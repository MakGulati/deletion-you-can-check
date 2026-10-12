"""Access metadata and cost composition.

The access table is the load-bearing part of Component C: it decides which
comparisons are permitted at all. A detector whose access set is understated
would be credited with beating a baseline it actually dominates in information,
which is precisely the claim the paper's central question exists to avoid making
by accident.

So the table is checked three ways: against the collector's real call graph,
against the coarse access classes the v2 preregistration already froze, and
against internal consistency (a detector needing a second checkpoint must list
the model, and vice versa). None of it needs a GPU.
"""

from __future__ import annotations

import math

import pytest

from deletion_signals.benchmark import (
    ACCESS,
    BASE,
    BOTH_CHECKPOINTS,
    CURVATURE,
    FINETUNED,
    PER_EXAMPLE_GRADIENTS,
    PER_EXAMPLE_LOSS,
    UNLEARNED,
    access_table,
    build_results,
    compose_costs,
    comparable,
)

BASELINE = "loss_ratio_vs_finetuned"


# ------------------------------------------------------------- the call graph
#
# collect.py's documented per-sample budget. Written out here independently, so
# a change to the collector that is not reflected in the access table fails
# rather than silently altering every cost in the paper.
EXPECTED_PASSES = {
    #                        (forward-only, backward)
    "loss_unlearned":         (1, 0),
    "loss_ratio":             (2, 0),
    "loss_delta":             (2, 0),
    "loss_ratio_vs_finetuned": (2, 0),
    "loss_delta_vs_finetuned": (2, 0),
    "grad_norm_sq":           (1, 1),
    "grad_ratio":             (2, 2),
    "log_grad_ratio":         (2, 2),
    "raw_cosine":             (1, 1),
    "alignment_cos_sq":       (1, 1),
    "sif":                    (1, 1),
    "combined_log_score":     (1, 1),
    "fuma_lora_ratio":        (1, 1),
}


def test_every_detector_declares_its_pass_budget():
    assert set(ACCESS) == set(EXPECTED_PASSES)
    for name, (forwards, backwards) in EXPECTED_PASSES.items():
        spec = ACCESS[name]
        assert spec.forward_passes_per_sample == forwards, name
        assert spec.backward_passes_per_sample == backwards, name


def test_a_backward_pass_counts_its_own_forward():
    """A backward is not free of its forward, and a table that pretends
    otherwise understates every gradient detector by half."""
    for spec in ACCESS.values():
        assert spec.forward_passes_per_sample >= spec.backward_passes_per_sample


def test_every_score_column_evaluated_by_the_study_has_an_access_row():
    from deletion_signals.schema import SCORE_COLUMNS

    missing = set(SCORE_COLUMNS) - set(ACCESS)
    assert not missing, f"score columns with no declared access: {sorted(missing)}"


# ------------------------------------------------------- internal consistency


def test_gradient_detectors_also_possess_the_per_example_loss():
    """A gradient cannot be computed without the loss it differentiates.

    Access is what the holder POSSESSES, not what the formula consumes; omitting
    this would let a gradient detector fail the subset test against a loss
    baseline it strictly dominates.
    """
    for spec in ACCESS.values():
        if PER_EXAMPLE_GRADIENTS in spec.access:
            assert PER_EXAMPLE_LOSS in spec.access, spec.name


def test_a_detector_reading_two_checkpoints_says_so_on_every_axis():
    for spec in ACCESS.values():
        if BOTH_CHECKPOINTS in spec.access:
            assert UNLEARNED in spec.access, spec.name
            assert FINETUNED in spec.access, spec.name
            assert "finetuned" in spec.models_resident, spec.name
        if FINETUNED in spec.access and UNLEARNED in spec.access:
            assert BOTH_CHECKPOINTS in spec.access, spec.name


def test_the_displacement_requires_both_checkpoints():
    for spec in ACCESS.values():
        if spec.needs_direction:
            assert BOTH_CHECKPOINTS in spec.access, spec.name


def test_curvature_implies_auxiliary_data():
    """A Fisher estimate needs held-out samples the auditor must possess and
    must keep disjoint from what it scores; that is an access requirement."""
    from deletion_signals.benchmark import AUXILIARY_DATA

    for spec in ACCESS.values():
        if CURVATURE in spec.access:
            assert AUXILIARY_DATA in spec.access, spec.name
            assert spec.needs_curvature, spec.name


def test_models_resident_matches_the_declared_pass_counts():
    for spec in ACCESS.values():
        if spec.forward_base:
            assert "base" in spec.models_resident, spec.name
        if spec.forward_finetuned or spec.backward_finetuned:
            assert "finetuned" in spec.models_resident, spec.name
        if spec.backward_shadow:
            assert "shadow_bf16" in spec.models_resident, spec.name


# ------------------------------------------------------ the matched-access rule


def test_a_detector_is_comparable_to_itself():
    for name in ACCESS:
        assert comparable(name, name)


def test_the_gradient_family_subsumes_the_two_checkpoint_loss_baseline():
    """These are the challengers the central question is about; if any of them
    were not comparable, that question could not be asked at all."""
    for challenger in ("raw_cosine", "alignment_cos_sq", "grad_ratio",
                       "log_grad_ratio", "combined_log_score"):
        assert comparable(challenger, BASELINE), challenger


def test_single_checkpoint_detectors_are_not_comparable_to_the_baseline():
    """grad_norm_sq, sif and fuma never read the pre-deletion checkpoint, so no
    advantage over the two-checkpoint baseline may be claimed for them."""
    for challenger in ("grad_norm_sq", "sif", "fuma_lora_ratio", "loss_unlearned"):
        assert not comparable(challenger, BASELINE), challenger


def test_base_calibrated_loss_is_not_comparable_to_the_finetuned_baseline():
    """Different information, not less: the base model is public, the
    pre-deletion checkpoint is not. Neither subsumes the other."""
    assert not comparable("loss_ratio", BASELINE)
    assert not comparable(BASELINE, "loss_ratio")


def test_the_baseline_is_subsumed_by_its_own_additive_form():
    assert comparable("loss_delta_vs_finetuned", BASELINE)
    assert comparable(BASELINE, "loss_delta_vs_finetuned")


def test_an_unknown_detector_raises_rather_than_defaulting():
    with pytest.raises(KeyError, match="Unknown detector"):
        comparable("no_such_detector", BASELINE)


# ------------------------------------- agreement with the v2 preregistration


COARSE_TO_FINE = {
    "unlearned": UNLEARNED,
    "base": BASE,
    "finetuned": FINETUNED,
    "gradients": PER_EXAMPLE_GRADIENTS,
    "curvature": CURVATURE,
    # `lora_structure` has no counterpart among the FLLM audit's enumerated
    # dimensions -- it describes knowledge about the parameter space rather than
    # an artifact the holder possesses -- so it is deliberately not mapped.
}


def test_the_fine_access_table_does_not_contradict_the_frozen_v2_classes():
    from deletion_signals.audit import Preregistration

    prereg = Preregistration.load()
    for detector, spec in ACCESS.items():
        coarse = prereg.access_of(detector)
        if not coarse:
            continue
        for entry in coarse:
            fine = COARSE_TO_FINE.get(entry)
            if fine is None:
                continue
            assert fine in spec.access, (
                f"{detector}: v2 declares access class {entry!r} but the fine "
                f"table omits {fine!r}"
            )


def test_the_primary_detector_is_the_matched_access_baseline():
    from deletion_signals.audit import Preregistration

    assert Preregistration.load().primary_detector == BASELINE


# -------------------------------------------------------------- composition


def _primitives():
    """Synthetic timings whose composed answers can be computed by hand."""
    def timing(per_item, peak=1000.0):
        return {"median": per_item * 10, "per_item_median": per_item,
                "n_items": 10, "peak_gpu_mib": peak}

    return {
        "n_samples": 10,
        "timings": {
            "load_pretrained_base": {"median": 5.0},
            "load_unlearned_adapter": {"median": 1.0},
            "load_finetuned_checkpoint": {"median": 6.0},
            "forward_unlearned": timing(0.010),
            "forward_finetuned": timing(0.010),
            "forward_base": timing(0.010),
            "backward_unlearned": timing(0.040, peak=3000.0),
            "backward_finetuned": timing(0.040, peak=3000.0),
            "build_direction_once": {"median": 0.5},
        },
    }


def _row(rows, name):
    return next(r for r in rows if r["detector"] == name)


def test_standalone_cost_sums_the_passes_the_detector_actually_makes():
    rows = compose_costs(_primitives(), n_samples=100)

    # two forward-only passes, 0.010 each
    assert _row(rows, BASELINE)["standalone_seconds_per_sample"] == pytest.approx(0.020)
    # one backward on the unlearned model
    assert _row(rows, "raw_cosine")["standalone_seconds_per_sample"] == pytest.approx(0.040)
    # two backward passes, one per checkpoint
    assert _row(rows, "grad_ratio")["standalone_seconds_per_sample"] == pytest.approx(0.080)


def test_marginal_cost_excludes_the_shared_unlearned_backward():
    """Four detectors come out of one backward pass; charging each of them the
    full pass would overstate the gradient family fourfold."""
    rows = compose_costs(_primitives(), n_samples=100)

    assert _row(rows, "raw_cosine")["marginal_seconds_per_sample"] == pytest.approx(0.0)
    assert _row(rows, "alignment_cos_sq")["marginal_seconds_per_sample"] == pytest.approx(0.0)
    assert _row(rows, "grad_norm_sq")["marginal_seconds_per_sample"] == pytest.approx(0.0)
    # grad_ratio still pays for the SECOND backward, on the fine-tuned model.
    assert _row(rows, "grad_ratio")["marginal_seconds_per_sample"] == pytest.approx(0.040)


def test_the_curvature_one_off_is_charged_to_the_detectors_that_need_it():
    rows = compose_costs(_primitives(), n_samples=100, curvature_rank=200)

    sif = _row(rows, "sif")
    assert sif["one_off_seconds"] == pytest.approx(5.0 + 1.0 + 200 * 0.040)
    assert any("Fisher curvature" in term for term in sif["one_off_terms"])
    assert _row(rows, "raw_cosine")["one_off_seconds"] < sif["one_off_seconds"]


def test_totals_scale_linearly_with_the_sample_count():
    small = compose_costs(_primitives(), n_samples=100)
    large = compose_costs(_primitives(), n_samples=200)

    a = _row(small, BASELINE)
    b = _row(large, BASELINE)
    assert b["standalone_seconds_total"] - a["standalone_seconds_total"] == pytest.approx(
        100 * a["standalone_seconds_per_sample"]
    )


def test_throughput_is_the_reciprocal_of_the_per_sample_cost():
    rows = compose_costs(_primitives(), n_samples=100)
    row = _row(rows, BASELINE)
    assert row["throughput_samples_per_second"] == pytest.approx(
        1.0 / row["standalone_seconds_per_sample"]
    )


def test_an_unmeasured_primitive_yields_no_number_rather_than_a_partial_sum():
    """FUMA needs the bf16 shadow. If that measurement is missing -- it does not
    always fit -- the cost must be absent, not a total silently missing a term."""
    primitives = _primitives()
    rows = compose_costs(primitives, n_samples=100)

    fuma = _row(rows, "fuma_lora_ratio")
    assert fuma["standalone_seconds_per_sample"] is None
    assert fuma["standalone_seconds_total"] is None
    assert fuma["unmeasured_terms"] is True


def test_the_document_builds_without_any_measurement():
    document = build_results(None, n_samples=800)
    assert document["matched_access_baseline"] == BASELINE
    assert len(document["access_table"]) == len(ACCESS)
    assert "costs" not in document
    assert document["central_comparison"]["all_challengers_comparable"] is True


def test_the_access_table_flags_comparability_on_every_row():
    for row in access_table():
        assert row["comparable_to_matched_access_baseline"] == comparable(
            row["detector"], BASELINE
        )
