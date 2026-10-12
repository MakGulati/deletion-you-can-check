"""The three-axis audit, on synthetic runs whose correct verdict is known.

Each fixture is built so the answer can be written down without running the
code: a population whose losses are fixed has a known perplexity ratio, a
detector that separates two populations perfectly has AUC 1.0, and an RNFF is a
ratio of two differences that are set explicitly.

The cases that matter most here are the ones a membership-only evaluation gets
wrong: a globally destroyed model must NOT be classified as successful, and a
detector that scores well on a control where nothing was deleted must not be
credited with finding deletion.
"""

from __future__ import annotations

import json
import math
import os

import pytest

from deletion_signals.audit import (
    EXACT_RETRAIN,
    NULL_REFIT,
    Preregistration,
    build_audit,
    classify,
    data_quality,
    detector_floors,
    population_losses,
    retrain_losses_by_seed,
    rnff,
)
from deletion_signals.audit_report import format_report
from deletion_signals.orientations import OrientationTable
from deletion_signals.schema import ScoreRow, load_any, make_flags, write_score_file


@pytest.fixture(scope="module")
def prereg() -> Preregistration:
    return Preregistration.load()


@pytest.fixture(scope="module")
def table() -> OrientationTable:
    return OrientationTable.load()


# ----------------------------------------------------------- preregistration


def test_the_shipped_preregistration_loads_and_declares_three_axes(prereg):
    assert prereg.raw["schema_version"] == 2
    assert prereg.membership["equivalence_region"] == [0.45, 0.55]
    assert prereg.utility["max_perplexity_ratio"] == 1.10
    assert prereg.selectivity["min_forget"] == 0.50
    assert prereg.selectivity["max_retain"] == 0.20


def test_the_membership_region_is_two_sided(prereg):
    """An AUC of 0.05 is not 'membership gone' -- it is an equally strong
    membership signal read in the other direction."""
    low, high = prereg.membership["equivalence_region"]
    assert low < 0.5 < high


def test_the_primary_detector_is_declared_with_its_access(prereg):
    assert prereg.primary_detector == "loss_ratio_vs_finetuned"
    assert set(prereg.access_of("loss_ratio_vs_finetuned")) == {"unlearned", "finetuned"}


def test_gradient_detectors_declare_strictly_more_access(prereg):
    """The matched-access rule needs this to be true, or the comparison it
    licenses would be meaningless."""
    assert prereg.weaker_or_equal_access("raw_cosine", "loss_ratio_vs_finetuned")
    assert not prereg.weaker_or_equal_access("loss_ratio_vs_finetuned", "raw_cosine")


def test_every_detector_in_the_preregistration_has_an_orientation(prereg, table):
    for detector in prereg.detectors:
        for task in table.task_names:
            assert table.orientation(detector, task).direction


def test_the_conflicts_of_interest_are_declared(prereg):
    """Both post-hoc choices must be on the record, or this is a description
    rather than a preregistration."""
    items = " ".join(c["item"] for c in prereg.raw["declared_conflicts"])
    assert "loss_ratio_vs_finetuned" in items
    assert "raw_cosine" in items


# --------------------------------------------------------------------- RNFF


def test_rnff_is_one_when_the_method_matches_exact_retraining():
    # Gold-standard forget move D = 0.4; the method moved forget by 0.4.
    assert rnff(loss_post=2.4, loss_finetuned=2.0, denominator=0.4) == pytest.approx(1.0)


def test_rnff_is_the_fraction_of_the_retraining_move_achieved():
    assert rnff(loss_post=2.1, loss_finetuned=2.0, denominator=0.4) == pytest.approx(0.25)


def test_rnff_is_zero_when_nothing_moved():
    assert rnff(loss_post=2.0, loss_finetuned=2.0, denominator=0.4) == pytest.approx(0.0)


def test_rnff_exceeds_one_when_the_method_overshoots():
    assert rnff(loss_post=2.8, loss_finetuned=2.0, denominator=0.4) == pytest.approx(2.0)


def test_rnff_is_absent_when_the_gold_standard_move_is_unmeasurable():
    """If exact retraining barely moved the forget set, the scale the whole
    criterion is expressed in does not exist for that seed."""
    assert rnff(loss_post=2.5, loss_finetuned=2.0, denominator=1e-7) is None


def test_rnff_is_absent_when_any_input_is_missing():
    assert rnff(None, 2.0, 0.4) is None
    assert rnff(2.1, None, 0.4) is None
    assert rnff(2.1, 2.0, None) is None


def test_the_denominator_is_the_forget_sets_gold_standard_move():
    from deletion_signals.audit import rnff_denominator

    assert rnff_denominator({"forget": 2.4, "retain": 2.0}, 2.0) == pytest.approx(0.4)
    assert rnff_denominator(None, 2.0) is None
    assert rnff_denominator({"retain": 2.0}, 2.0) is None


def test_retain_is_normalised_by_the_same_denominator_as_forget():
    """Not by its own retraining change, which is ~0 by construction because
    exact retraining trains on D_retain. A specificity clause built on that
    would never fire -- including for a destroyed model."""
    denominator = 0.4
    # A destroyed model moves retain by 10 nats: 25x the intended effect.
    assert rnff(12.0, 2.0, denominator) == pytest.approx(25.0)
    # A specific method moves retain by 0.003: under a hundredth of it.
    assert rnff(2.003, 2.0, denominator) == pytest.approx(0.0075)


# ------------------------------------------------------------ synthetic runs


def _rows(seed, method, *, post, finetuned, base, cosine, loss_vs_ft, n=20):
    """One score table with per-population losses fixed to the values given."""
    rows = []
    for population in ("forget", "retain", "test"):
        for index in range(n):
            jitter = (index - n / 2) * 1e-6
            unlearned = post[population] + jitter
            rows.append(
                ScoreRow(
                    seed=seed, method=method, dataset="synthetic",
                    model_id="synthetic", sample_id=f"{population}:{index}",
                    text_sha1=f"{population}{index:04d}", population=population,
                    n_tokens=100,
                    loss_unlearned=unlearned,
                    loss_base=base,
                    loss_ratio=unlearned / base,
                    loss_delta=unlearned - base,
                    loss_finetuned=finetuned[population],
                    # Set independently of the losses so a fixture can dial the
                    # detector and the utility axis apart from each other.
                    loss_ratio_vs_finetuned=loss_vs_ft[population] + jitter,
                    loss_delta_vs_finetuned=loss_vs_ft[population] + jitter,
                    grad_norm_sq=1.0,
                    raw_cosine=cosine[population] + jitter,
                    alignment_cos_sq=(cosine[population] + jitter) ** 2,
                    flags=make_flags(),
                )
            )
    return rows


FLAT = {"forget": 2.0, "retain": 2.0, "test": 2.4}
SEPARATING = {"forget": 0.9, "retain": -0.9, "test": 0.0}
CHANCE = {"forget": 0.0, "retain": 0.0, "test": 0.0}
NO_DETECTION = {"forget": 1.0, "retain": 1.0, "test": 1.0}
DETECTION = {"forget": 1.5, "retain": 0.5, "test": 1.0}


def _write(tmp_path, seed, method, label=None, **kwargs):
    name = f"{method}_{label or ''}_seed{seed}.json".replace("__", "_")
    path = os.path.join(tmp_path, name)
    write_score_file(
        path,
        _rows(seed, method, **kwargs),
        {
            "seed": seed,
            "method": method,
            "unlearning_run": {
                "hyperparameters": {"ascent_lr": 2e-5},
                "steps_taken": 100,
                "stop_reason": "completed",
            },
        },
    )
    return path


def _control(tmp_path, seed, method, forget_loss):
    """A control whose forget loss is set explicitly: exact retraining raises it
    to the non-member level, the null refit leaves it where it was."""
    return _write(
        tmp_path, seed, method,
        post={"forget": forget_loss, "retain": 2.0, "test": 2.4},
        finetuned=FLAT, base=3.0, cosine=CHANCE, loss_vs_ft=NO_DETECTION,
    )


def test_the_retrain_control_supplies_the_rnff_denominator(tmp_path, table, prereg):
    paths = [_control(tmp_path, s, EXACT_RETRAIN, 2.4) for s in (42, 43, 44)]
    files = [load_any(p) for p in paths]
    losses = retrain_losses_by_seed(files)
    assert set(losses) == {42, 43, 44}
    assert losses[42]["forget"] == pytest.approx(2.4, abs=1e-3)


def test_the_denominator_is_taken_per_seed_not_pooled(tmp_path, table, prereg):
    """Seed 42's methods must be normalised by seed 42's own control: the two
    share a split and a starting checkpoint, and another seed's denominator was
    measured on different data."""
    paths = [
        _control(tmp_path, 42, EXACT_RETRAIN, 2.4),
        _control(tmp_path, 43, EXACT_RETRAIN, 2.8),
    ]
    losses = retrain_losses_by_seed([load_any(p) for p in paths])
    assert losses[42]["forget"] != losses[43]["forget"]


# ------------------------------------------------------------ the three axes


def _study(tmp_path, method_post, method_loss_vs_ft, *, seeds=(42, 43, 44),
           method="ga_gd", retrain_forget=2.4):
    paths = []
    for seed in seeds:
        paths.append(_control(tmp_path, seed, EXACT_RETRAIN, retrain_forget))
        paths.append(_control(tmp_path, seed, NULL_REFIT, 2.0))
        paths.append(
            _write(
                tmp_path, seed, method, label="m",
                post=method_post, finetuned=FLAT, base=3.0,
                cosine=CHANCE, loss_vs_ft=method_loss_vs_ft,
            )
        )
    return [load_any(p) for p in paths]


def test_a_globally_destroyed_model_is_not_certified_as_successful(tmp_path, table, prereg):
    """The failure mode the whole audit exists to prevent. Membership evidence
    is gone -- but only because every population's loss exploded together."""
    destroyed = {"forget": 12.0, "retain": 12.0, "test": 12.0}
    files = _study(tmp_path, destroyed, DETECTION)
    results = build_audit(files, table, prereg, n_bootstrap=200)

    setting = next(s for s in results["settings"] if s["method"] == "ga_gd")
    met = setting["axes_met_seed_counts"]
    assert met["membership_removed"] == 3      # forget and test now equal
    assert met["utility_preserved"] == 0       # retain perplexity exploded
    assert setting["candidate_operating_point"] is False
    assert setting["unanimous_classification"] == "global_destruction"
    # The study-level statement must name the trap, not merely decline to
    # certify: a membership-only evaluation would have passed this run.
    outcome = results["verdict"]["outcome"].lower()
    assert "falsely certified" in outcome
    assert "utility exceeded the damage threshold" in outcome


def test_destruction_fails_the_selectivity_clause_by_construction(tmp_path, table, prereg):
    """Destruction moves retain as far as forget, so RNFF_retain rises with
    RNFF_forget -- which is exactly what the forget/retain ratio could not see."""
    destroyed = {"forget": 12.0, "retain": 12.0, "test": 12.0}
    files = _study(tmp_path, destroyed, DETECTION)
    results = build_audit(files, table, prereg, n_bootstrap=200)
    run = next(r for r in results["runs"] if r["method"] == "ga_gd")

    assert run["rnff"]["forget"] > 1.0
    assert run["axes"]["selective_forgetting"]["met"] is False


def test_a_useful_model_that_still_leaks_membership_is_incomplete_forgetting(
    tmp_path, table, prereg
):
    barely = {"forget": 2.05, "retain": 2.0, "test": 2.4}
    files = _study(tmp_path, barely, DETECTION)
    results = build_audit(files, table, prereg, n_bootstrap=200)

    setting = next(s for s in results["settings"] if s["method"] == "ga_gd")
    met = setting["axes_met_seed_counts"]
    assert met["utility_preserved"] == 3
    assert met["membership_removed"] == 0
    assert met["selective_forgetting"] == 0     # RNFF ~0.125, below 0.50
    assert setting["unanimous_classification"] == "incomplete_forgetting"


def test_a_genuine_operating_point_with_a_fingerprint_is_recognised(
    tmp_path, table, prereg
):
    """Forget loss raised to the non-member level, retain untouched, and the
    primary detector separating forget from retain."""
    succeeded = {"forget": 2.4, "retain": 2.0, "test": 2.4}
    files = _study(tmp_path, succeeded, DETECTION)
    results = build_audit(files, table, prereg, n_bootstrap=300)

    setting = next(s for s in results["settings"] if s["method"] == "ga_gd")
    assert setting["axes_met_seed_counts"]["membership_removed"] == 3
    assert setting["axes_met_seed_counts"]["utility_preserved"] == 3
    assert setting["axes_met_seed_counts"]["selective_forgetting"] == 3
    assert setting["candidate_operating_point"] is True
    assert setting["unanimous_classification"] == "persistent_fingerprint"


def test_deletion_at_chance_after_success_is_its_own_classification(
    tmp_path, table, prereg
):
    succeeded = {"forget": 2.4, "retain": 2.0, "test": 2.4}
    files = _study(tmp_path, succeeded, NO_DETECTION)
    results = build_audit(files, table, prereg, n_bootstrap=300)
    setting = next(s for s in results["settings"] if s["method"] == "ga_gd")
    assert setting["unanimous_classification"] == "successful_deletion_no_fingerprint"


# ------------------------------------------------------- the null-refit floor


def test_a_detector_must_clear_the_null_refit_floor(tmp_path, table, prereg):
    """A detector that separates forget from retain on a run where NOTHING was
    deleted is reading training randomness. It must not be credited."""
    files = []
    for seed in (42, 43, 44):
        files.append(load_any(_control(tmp_path, seed, EXACT_RETRAIN, 2.4)))
        # The null refit deletes nothing, yet its detector separates perfectly.
        files.append(load_any(_write(
            tmp_path, seed, NULL_REFIT, label="leaky",
            post={"forget": 2.0, "retain": 2.0, "test": 2.4},
            finetuned=FLAT, base=3.0, cosine=CHANCE, loss_vs_ft=DETECTION,
        )))
        files.append(load_any(_write(
            tmp_path, seed, "ga_gd", label="m",
            post={"forget": 2.4, "retain": 2.0, "test": 2.4},
            finetuned=FLAT, base=3.0, cosine=CHANCE, loss_vs_ft=DETECTION,
        )))

    results = build_audit(files, table, prereg, n_bootstrap=300)
    floors = results["metadata"]["null_refit_floors"]
    assert floors[prereg.primary_detector]["mean"] == pytest.approx(1.0, abs=0.02)

    run = next(r for r in results["runs"] if r["method"] == "ga_gd")
    deletion = run["axes"]["deletion_detectable"]
    assert deletion["clears_thresholds"] is True      # AUC is 1.0
    assert deletion["clears_null_floor"] is False     # but so is the null's
    assert deletion["met"] is False


def test_without_a_null_control_the_deletion_axis_refuses_rather_than_passes(
    tmp_path, table, prereg
):
    """An unverified detector is not a verified one. Refusing is conservative;
    passing would credit a signal whose false-positive rate is unmeasured."""
    files = [load_any(_control(tmp_path, s, EXACT_RETRAIN, 2.4)) for s in (42, 43, 44)]
    for seed in (42, 43, 44):
        files.append(load_any(_write(
            tmp_path, seed, "ga_gd", label="m",
            post={"forget": 2.4, "retain": 2.0, "test": 2.4},
            finetuned=FLAT, base=3.0, cosine=CHANCE, loss_vs_ft=DETECTION,
        )))

    results = build_audit(files, table, prereg, n_bootstrap=200)
    run = next(r for r in results["runs"] if r["method"] == "ga_gd")
    assert run["axes"]["deletion_detectable"]["clears_thresholds"] is True
    assert run["axes"]["deletion_detectable"]["met"] is False


# ------------------------------------------------------------- data quality


def test_data_quality_flags_overlapping_populations():
    rows = [
        {"population": "forget", "text_sha1": "dup", "flags": {}},
        {"population": "retain", "text_sha1": "dup", "flags": {}},
        {"population": "test", "text_sha1": "x", "flags": {}},
    ]
    quality = data_quality(rows)
    assert quality["populations_disjoint"] is False
    assert quality["population_overlaps"]["forget&retain"] == 1


def test_data_quality_counts_nulls_and_truncation():
    rows = [
        {"population": "forget", "text_sha1": "a", "loss_ratio": None,
         "flags": {"truncated": True}},
        {"population": "retain", "text_sha1": "b", "loss_ratio": float("nan"),
         "flags": {"truncated": False}},
    ]
    quality = data_quality(rows)
    assert quality["null_counts"]["loss_ratio"] == 1
    assert quality["nonfinite_counts"]["loss_ratio"] == 1
    assert quality["truncated"] == 1


def test_failed_runs_are_not_excluded_from_the_tables(tmp_path, table, prereg):
    """A collapsed run is a data point. Dropping it would make every summary a
    survivorship statistic."""
    destroyed = {"forget": 12.0, "retain": 12.0, "test": 12.0}
    files = _study(tmp_path, destroyed, DETECTION)
    results = build_audit(files, table, prereg, n_bootstrap=200)
    methods = [r for r in results["runs"] if r["method"] == "ga_gd"]
    assert len(methods) == 3
    report = format_report(results, prereg)
    assert "Data quality" in report


# ------------------------------------------------------------------- report


def test_the_report_states_thresholds_access_and_caveats(tmp_path, table, prereg):
    files = _study(tmp_path, {"forget": 2.4, "retain": 2.0, "test": 2.4}, DETECTION)
    results = build_audit(files, table, prereg, n_bootstrap=200)
    report = format_report(results, prereg)

    assert "white-box deletion-audit" in report
    assert "ArxivMIA only" in report
    assert "n = 3 seeds" in report
    assert "null refit" in report.lower()
    assert "RNFF" in report
    # Every seed is shown, not just the aggregate.
    assert "## Per seed" in report
    for seed in (42, 43, 44):
        assert f"| {seed} |" in report


def test_the_results_json_is_strict(tmp_path, table, prereg):
    from deletion_signals.audit import jsonable

    files = _study(tmp_path, {"forget": 2.4, "retain": 2.0, "test": 2.4}, DETECTION)
    results = build_audit(files, table, prereg, n_bootstrap=200)
    encoded = json.dumps(jsonable(results), allow_nan=False)
    assert "NaN" not in encoded and "Infinity" not in encoded


# ------------------------------------------------- grouping must not pool arms


def test_stabilisation_arms_are_not_pooled_by_learning_rate():
    """A stabilisation study holds the learning rate FIXED and varies which
    constraint is active. Keying the group on the learning rate would average
    five different interventions into one row and destroy the comparison."""
    from deletion_signals.audit import setting_label

    base = {"hyperparameters": {"ascent_lr": 5e-5}}
    off = {"retain_kl_weight": 0.0, "trust_region_radius": None,
           "max_grad_norm": 1.0, "utility_guard_ratio": None}

    labels = {
        setting_label({"unlearning_run": {**base, "stabilizers": {**off}}}, "ga_gd_stabilized"),
        setting_label({"unlearning_run": {**base, "stabilizers": {**off, "retain_kl_weight": 1.0}}}, "ga_gd_stabilized"),
        setting_label({"unlearning_run": {**base, "stabilizers": {**off, "trust_region_radius": 0.747}}}, "ga_gd_stabilized"),
        setting_label({"unlearning_run": {**base, "stabilizers": {**off, "max_grad_norm": 0.1}}}, "ga_gd_stabilized"),
        setting_label({"unlearning_run": {**base, "stabilizers": {**off, "utility_guard_ratio": 1.05}}}, "ga_gd_stabilized"),
    }
    assert len(labels) == 5, f"arms collapsed into {len(labels)} group(s): {labels}"
    assert any("unconstrained" in l for l in labels)


def test_scrub_configs_are_distinguished_by_both_swept_knobs():
    from deletion_signals.audit import setting_label

    labels = {
        setting_label({"unlearning_run": {"hyperparameters":
            {"forget_kl_weight": kl, "learning_rate": lr}}}, "scrub")
        for kl in (1.0, 4.0) for lr in (1e-5, 5e-5)
    }
    assert len(labels) == 4, f"configs collapsed into {len(labels)}: {labels}"
