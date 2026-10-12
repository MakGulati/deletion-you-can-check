"""Low-prevalence metrics, against synthetic scores with known behaviour.

The failure modes this file exists to catch are all silent ones. A precision
that reads 1.0 because no false positive happened to appear in 600 draws; a TPR
at an FPR the sample cannot resolve, reported as though it were measured; a
prevalence adjustment that quietly moves recall as well as precision. Each of
those produces a number that looks like a result, so each gets a test built on
scores whose answer is known in advance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from deletion_signals.lowprev import (
    DEFAULT_PREVALENCES,
    aggregate,
    attach_prevalence,
    evaluate_detector,
    fpr_resolution,
    operating_point_at_fpr,
    operating_point_at_top_k,
    parse_prevalence,
    prevalence_adjusted_precision,
    resampled_precision,
)
from deletion_signals.orientations import OrientationTable


# ----------------------------------------------------------------- prevalence


def test_prevalence_labels_parse_as_base_rates():
    assert parse_prevalence("1:1") == pytest.approx(0.5)
    assert parse_prevalence("1:10") == pytest.approx(1 / 11)
    assert parse_prevalence("1:100") == pytest.approx(1 / 101)
    assert parse_prevalence("0.25") == pytest.approx(0.25)


def test_a_malformed_prevalence_is_rejected():
    with pytest.raises(ValueError):
        parse_prevalence("0:5")
    with pytest.raises(ValueError):
        parse_prevalence("1.5")


def test_prevalence_adjusted_precision_matches_the_closed_form():
    """precision = pi*TPR / (pi*TPR + (1-pi)*FPR), computed by hand."""
    tpr, fpr, pi = 0.9, 0.05, 1 / 101
    expected = (pi * tpr) / (pi * tpr + (1 - pi) * fpr)
    assert prevalence_adjusted_precision(tpr, fpr, pi) == pytest.approx(expected)
    # Sanity on the magnitude: a 5% FPR at 1:100 is mostly false positives.
    assert prevalence_adjusted_precision(tpr, fpr, pi) < 0.2


def test_precision_falls_with_prevalence_while_recall_does_not():
    """The asymmetry that is the entire point of the component."""
    tpr, fpr = 0.95, 0.01
    precisions = [
        prevalence_adjusted_precision(tpr, fpr, parse_prevalence(label))
        for label in ("1:1", "1:10", "1:100")
    ]
    assert precisions == sorted(precisions, reverse=True)
    assert precisions[0] > 0.98
    assert precisions[-1] < 0.55


def test_a_perfect_detector_keeps_perfect_precision_at_every_prevalence():
    for label in DEFAULT_PREVALENCES:
        value = prevalence_adjusted_precision(1.0, 0.0, parse_prevalence(label))
        assert value == pytest.approx(1.0)


def test_precision_is_undefined_when_nothing_is_flagged():
    assert prevalence_adjusted_precision(0.0, 0.0, 0.5) is None


def test_an_out_of_range_prevalence_is_rejected():
    with pytest.raises(ValueError, match="prevalence must lie"):
        prevalence_adjusted_precision(0.9, 0.1, 1.5)


# ------------------------------------------------------------ FPR resolution


def test_fpr_resolution_is_one_over_the_negative_count():
    assert fpr_resolution(600) == pytest.approx(1 / 600)
    assert fpr_resolution(1000) == pytest.approx(0.001)
    assert math.isnan(fpr_resolution(0))


def test_an_unresolvable_fpr_target_returns_no_number():
    """600 negatives cannot express 0.1% FPR; the row must say so, not guess."""
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(3, 1, 200), rng.normal(0, 1, 600)])
    labels = [1] * 200 + [0] * 600

    point = operating_point_at_fpr(values, labels, 0.001)
    assert point.resolvable is False
    assert point.tpr is None
    assert point.precision is None
    assert "unresolvable" in point.note

    resolvable = operating_point_at_fpr(values, labels, 0.01)
    assert resolvable.resolvable is True
    assert resolvable.tpr is not None


def test_a_zero_observed_fpr_is_floored_rather_than_called_perfect():
    """Otherwise precision reads 1.0 at 1:100 on the strength of seeing no
    false positive in 600 draws, which the sample cannot support."""
    # Perfectly separated: every positive outranks every negative.
    values = [10.0] * 50 + [0.0] * 600
    labels = [1] * 50 + [0] * 600

    point = operating_point_at_top_k(values, labels, k=50)
    assert point.fpr == 0.0
    assert point.resolution_limited is True
    assert point.fpr_used == pytest.approx(1 / 600)
    assert "floored" in point.note

    attach_prevalence(point, ["1:100"])
    precision = point.prevalence["1:100"]["precision"]
    assert precision is not None
    assert precision < 1.0            # not certified perfect
    assert precision > 0.8            # but still very good


# ------------------------------------------------------------- operating points


def test_top_k_on_a_perfect_ranking_recovers_everything():
    values = list(range(100, 0, -1))          # descending
    labels = [1] * 30 + [0] * 70

    point = operating_point_at_top_k(values, labels)
    assert point.n_selected == 30
    assert point.recall == pytest.approx(1.0)
    assert point.precision == pytest.approx(1.0)


def test_top_k_precision_and_recall_coincide_at_the_positive_count():
    rng = np.random.default_rng(3)
    values = np.concatenate([rng.normal(1.0, 1, 40), rng.normal(0, 1, 160)])
    labels = [1] * 40 + [0] * 160

    point = operating_point_at_top_k(values, labels)
    assert point.precision == pytest.approx(point.recall)
    assert point.n_selected == 40


def test_top_k_does_not_over_select_when_scores_tie():
    """A constant score must still select exactly k, not the whole population."""
    values = [1.0] * 200
    labels = [1] * 50 + [0] * 150

    point = operating_point_at_top_k(values, labels)
    assert point.n_selected == 50


def test_a_random_detector_recovers_about_the_base_rate():
    rng = np.random.default_rng(7)
    values = rng.normal(0, 1, 800)
    labels = [1] * 200 + [0] * 600

    point = operating_point_at_top_k(values, labels)
    assert point.precision == pytest.approx(0.25, abs=0.08)


def test_degenerate_labels_produce_no_operating_point():
    point = operating_point_at_top_k([1.0, 2.0, 3.0], [1, 1, 1])
    assert point.resolvable is False
    assert point.precision is None


# ------------------------------------------------- the resampling cross-check


def test_the_resampling_check_agrees_with_the_analytic_formula():
    """The analytic value is authoritative; this proves it is also correct."""
    rng = np.random.default_rng(11)
    positives = rng.normal(3.0, 1.0, 200)
    negatives = rng.normal(0.0, 1.0, 600)
    values = np.concatenate([positives, negatives])
    labels = [1] * 200 + [0] * 600

    point = operating_point_at_top_k(values, labels)
    attach_prevalence(point, ["1:10"], values, labels, rng=np.random.default_rng(0),
                      resample=True)

    analytic = point.prevalence["1:10"]["precision"]
    simulated = point.prevalence["1:10"]["resampled"]["mean"]
    assert simulated == pytest.approx(analytic, abs=0.05)


# --------------------------------------------------- orientation and end-to-end


def _rows(forget_values, retain_values, column="loss_unlearned"):
    rows = []
    for index, value in enumerate(forget_values):
        rows.append({"population": "forget", "sample_id": f"forget:{index}",
                     "text_sha1": f"f{index}", column: value})
    for index, value in enumerate(retain_values):
        rows.append({"population": "retain", "sample_id": f"retain:{index}",
                     "text_sha1": f"r{index}", column: value})
    return rows


def test_evaluate_detector_reports_every_required_metric():
    rng = np.random.default_rng(5)
    rows = _rows(rng.normal(2.0, 1.0, 200), rng.normal(0.0, 1.0, 600))
    table = OrientationTable.load()

    entry = evaluate_detector(rows, "loss_unlearned", table)

    assert entry["n_positive"] == 200
    assert entry["n_negative"] == 600
    assert 0.85 < entry["roc_auc"] < 1.0
    assert entry["average_precision"] is not None
    assert entry["tpr_at_fpr"]["0.01"]["resolvable"] is True
    assert entry["tpr_at_fpr"]["0.001"]["resolvable"] is False
    assert entry["tpr_at_fpr"]["0.001"]["tpr"] is None
    assert entry["k"] == 200
    assert entry["precision_at_deletion_set_size"] is not None
    assert entry["recall_at_deletion_set_size"] == entry["top_k_recall"]
    for label in DEFAULT_PREVALENCES:
        assert label in entry["operating_points"]["top_200"]["prevalence"]


def test_an_absent_detector_column_is_absent_not_imputed():
    """The confirmatory GA+GD runs really do lack sif and fuma; a zero there
    would enter the rankings as a measurement."""
    rows = _rows([1.0] * 10, [0.0] * 10)
    table = OrientationTable.load()
    assert evaluate_detector(rows, "sif", table) is None


def test_the_preregistered_orientation_is_applied_not_optimised():
    """A detector that is anti-predictive under its declared direction must
    report an AUC below 0.5, never max(auc, 1-auc)."""
    rows = _rows([0.0] * 100, [5.0] * 300)   # forget scores LOWER than retain
    table = OrientationTable.load()

    entry = evaluate_detector(rows, "loss_unlearned", table)
    assert entry["direction"] == "higher_is_positive"
    assert entry["roc_auc"] < 0.05


def test_evaluation_is_deterministic():
    rng = np.random.default_rng(9)
    rows = _rows(rng.normal(1.5, 1, 120), rng.normal(0, 1, 400))
    table = OrientationTable.load()

    first = evaluate_detector(rows, "loss_unlearned", table)
    second = evaluate_detector(rows, "loss_unlearned", table)
    assert first == second


def test_non_finite_values_are_dropped_and_counted():
    rows = _rows([1.0, float("nan"), 2.0], [0.0, 0.0, 0.0])
    table = OrientationTable.load()

    entry = evaluate_detector(rows, "loss_unlearned", table)
    assert entry["n_dropped"] == 1
    assert entry["n_positive"] == 2


# ------------------------------------------------------------- aggregation


def _evaluation(seed, method="npo", setting="lr=2.8e-05, beta=0.1", auc=0.9):
    return {
        "path": f"/tmp/{seed}.json",
        "method": method,
        "setting": setting,
        "seed": seed,
        "populations": {},
        "detectors": {
            "loss_unlearned": {
                "roc_auc": auc,
                "average_precision": auc - 0.1,
                "precision_at_deletion_set_size": auc - 0.2,
                "recall_at_deletion_set_size": auc - 0.2,
                "top_k_recall": auc - 0.2,
                "tpr_at_fpr": {"0.01": {"tpr": 0.5}, "0.001": {"tpr": None}},
                "operating_points": {
                    "top_200": {
                        "prevalence": {
                            "1:1": {"precision": 0.9},
                            "1:10": {"precision": 0.7},
                            "1:100": {"precision": 0.2},
                        }
                    }
                },
            }
        },
        "detectors_not_collected": [],
    }


def test_aggregation_summarises_over_seeds_not_samples():
    evaluations = [
        _evaluation(42, auc=0.90),
        _evaluation(43, auc=0.92),
        _evaluation(44, auc=0.94),
    ]
    result = aggregate(evaluations)
    block = result["settings"][0]["detectors"]["loss_unlearned"]

    assert block["n_seeds"] == 3
    assert block["seeds"] == [42, 43, 44]
    assert block["roc_auc"]["mean"] == pytest.approx(0.92)
    assert block["roc_auc"]["n_seeds"] == 3
    assert block["roc_auc"]["per_seed"] == {42: 0.90, 43: 0.92, 44: 0.94}
    assert block["complete_across_seeds"] is True


def test_settings_are_not_pooled_across_different_hyperparameters():
    evaluations = [
        _evaluation(42, setting="lr=1e-05, beta=0.1"),
        _evaluation(42, setting="lr=0.0001, beta=0.1"),
    ]
    result = aggregate(evaluations)
    assert len(result["settings"]) == 2


def test_an_unresolvable_metric_does_not_enter_the_mean():
    evaluations = [_evaluation(42), _evaluation(43), _evaluation(44)]
    block = aggregate(evaluations)["settings"][0]["detectors"]["loss_unlearned"]
    assert block["tpr_at_fpr_0.001"]["n_seeds"] == 0
    assert block["tpr_at_fpr_0.001"]["mean"] is None


def test_a_single_seed_reports_no_standard_deviation():
    block = aggregate([_evaluation(42)])["settings"][0]["detectors"]["loss_unlearned"]
    assert block["n_seeds"] == 1
    assert math.isnan(block["roc_auc"]["sd"])
