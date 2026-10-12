"""Metrics, intervals and the across-seed comparison."""

from __future__ import annotations

import math

import numpy as np
import pytest

from deletion_signals.reproducibility import new_rng
from deletion_signals.stats import (
    across_seed_comparison,
    average_precision,
    bootstrap_auc_ci,
    finitize,
    paired_bootstrap_auc_diff,
    permutation_test_auc_above_chance,
    roc_auc,
    spearman,
    summarise_across_seeds,
    tpr_at_fpr,
)


# -------------------------------------------------------------------- finitize


def test_finitize_puts_minus_inf_below_every_finite_score():
    out = finitize([float("-inf"), 1.0, 2.0])
    assert out[0] < out[1] < out[2]
    assert np.isfinite(out).all()


def test_finitize_preserves_ranking():
    scores = [float("-inf"), -5.0, 0.0, 3.0]
    out = finitize(scores)
    assert list(np.argsort(out)) == [0, 1, 2, 3]


def test_finitize_leaves_nan_alone_so_it_fails_loudly():
    out = finitize([float("nan"), 1.0])
    assert math.isnan(out[0])


def test_finitize_handles_all_minus_inf():
    out = finitize([float("-inf")] * 3)
    assert np.isfinite(out).all()


# ------------------------------------------------------------------------ AUC


def test_perfect_separation_gives_auc_one():
    assert roc_auc([3.0, 4.0, 5.0, 0.0, 1.0, 2.0], [1, 1, 1, 0, 0, 0]) == pytest.approx(1.0)


def test_perfect_inversion_gives_auc_zero():
    assert roc_auc([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [1, 1, 1, 0, 0, 0]) == pytest.approx(0.0)


def test_all_ties_give_auc_half():
    assert roc_auc([1.0] * 6, [1, 1, 1, 0, 0, 0]) == pytest.approx(0.5)


def test_single_class_gives_nan_not_a_number_that_looks_real():
    assert math.isnan(roc_auc([1.0, 2.0], [1, 1]))


def test_average_precision_is_one_under_perfect_separation():
    assert average_precision([2.0, 3.0, 0.0, 1.0], [1, 1, 0, 0]) == pytest.approx(1.0)


# --------------------------------------------------------------- TPR at an FPR


def test_tpr_at_fpr_is_one_under_perfect_separation():
    values = [10.0] * 50 + [0.0] * 200
    labels = [1] * 50 + [0] * 200
    result = tpr_at_fpr(values, labels, 0.05)
    assert result.resolvable
    assert result.tpr == pytest.approx(1.0)


def test_tpr_at_a_target_finer_than_one_over_n_is_marked_unresolvable():
    """200 negatives cannot express a 0.1% FPR; reporting a number would be noise."""
    values = list(np.linspace(1, 0, 250))
    labels = [1] * 50 + [0] * 200
    result = tpr_at_fpr(values, labels, 0.001)
    assert result.resolvable is False
    assert "unresolvable" in result.note
    assert result.as_dict()["tpr"] is None


def test_tpr_at_a_reachable_target_is_reported():
    values = list(np.linspace(1, 0, 250))
    labels = [1] * 50 + [0] * 200
    result = tpr_at_fpr(values, labels, 0.05)
    assert result.resolvable
    assert 0.0 <= result.tpr <= 1.0


def test_tpr_reports_the_negative_count_alongside_the_number():
    result = tpr_at_fpr([1.0, 0.0], [1, 0], 0.5)
    assert result.n_negative == 1


# --------------------------------------------------------------- bootstrap CI


def test_bootstrap_point_estimate_is_the_observed_auc_not_the_resampling_mean():
    values = [3.0, 4.0, 5.0, 0.0, 1.0, 2.0]
    labels = [1, 1, 1, 0, 0, 0]
    estimate = bootstrap_auc_ci(values, labels, n_bootstrap=200, rng=new_rng(0))
    assert estimate.auc == pytest.approx(roc_auc(values, labels))


def test_bootstrap_interval_brackets_the_estimate_on_noisy_data():
    rng = np.random.default_rng(3)
    values = list(rng.normal(1.0, 1.0, 120)) + list(rng.normal(0.0, 1.0, 120))
    labels = [1] * 120 + [0] * 120
    estimate = bootstrap_auc_ci(values, labels, n_bootstrap=400, rng=new_rng(1))
    assert estimate.ci_lower <= estimate.auc <= estimate.ci_upper
    assert estimate.n_positive == 120 and estimate.n_negative == 120


def test_bootstrap_is_reproducible_for_a_given_rng_seed():
    values = list(np.linspace(0, 1, 60))
    labels = [1] * 30 + [0] * 30
    a = bootstrap_auc_ci(values, labels, n_bootstrap=100, rng=new_rng(7))
    b = bootstrap_auc_ci(values, labels, n_bootstrap=100, rng=new_rng(7))
    assert (a.ci_lower, a.ci_upper) == (b.ci_lower, b.ci_upper)


def test_bootstrap_on_a_single_class_returns_nan_rather_than_raising():
    estimate = bootstrap_auc_ci([1.0, 2.0], [1, 1], n_bootstrap=10)
    assert math.isnan(estimate.auc)


# ------------------------------------------------------- paired, within a run


def test_paired_bootstrap_recovers_the_observed_difference():
    labels = [1] * 40 + [0] * 40
    good = list(np.linspace(1, 2, 40)) + list(np.linspace(0, 1, 40))
    bad = list(np.linspace(0, 1, 40)) + list(np.linspace(1, 2, 40))
    result = paired_bootstrap_auc_diff(good, bad, labels, n_bootstrap=100, rng=new_rng(0))
    assert result["diff"] == pytest.approx(result["auc_a"] - result["auc_b"])
    assert result["diff"] > 0
    assert "within-run only" in result["caveat"]


def test_paired_bootstrap_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="aligned"):
        paired_bootstrap_auc_diff([1.0, 2.0], [1.0], [1, 0])


# ------------------------------------------------------- paired, across seeds


def test_across_seed_comparison_uses_only_shared_seeds():
    result = across_seed_comparison(
        "a", "b", {42: 0.7, 43: 0.8, 99: 0.9}, {42: 0.6, 43: 0.6}
    )
    assert result.seeds == [42, 43]
    assert result.mean_diff == pytest.approx(0.15)


def test_across_seed_comparison_uses_a_t_test_below_six_seeds_and_says_why():
    result = across_seed_comparison(
        "a", "b", {42: 0.70, 43: 0.72, 44: 0.68}, {42: 0.60, 43: 0.61, 44: 0.59}
    )
    assert result.test_name == "paired t-test"
    assert "Wilcoxon" in result.note
    assert result.mean_diff == pytest.approx(0.10, abs=1e-9)
    assert result.as_dict()["unit"].startswith("seed")


def test_across_seed_comparison_uses_wilcoxon_from_six_seeds():
    a = {40: 0.70, 41: 0.74, 42: 0.69, 43: 0.77, 44: 0.72, 45: 0.75}
    b = {40: 0.61, 41: 0.66, 42: 0.58, 43: 0.71, 44: 0.60, 45: 0.69}
    result = across_seed_comparison("a", "b", a, b)
    assert result.test_name == "Wilcoxon signed-rank"
    assert result.mean_diff > 0


def test_a_constant_difference_across_seeds_gets_no_test():
    """Zero between-seed variation cannot support a p-value, and a paired test
    fed that data manufactures one."""
    a = {1: 0.7, 2: 0.8, 3: 0.9}
    b = {1: 0.6, 2: 0.7, 3: 0.8}
    result = across_seed_comparison("a", "b", a, b)
    assert result.test_name == "none"
    assert result.mean_diff == pytest.approx(0.1)
    assert "no between-seed variation" in result.note
    assert math.isnan(result.p_value)


def test_across_seed_comparison_declines_to_test_a_single_seed():
    result = across_seed_comparison("a", "b", {42: 0.7}, {42: 0.6})
    assert result.test_name == "none"
    assert "no test performed" in result.note
    assert math.isnan(result.sd_diff)


def test_across_seed_comparison_handles_identical_scores():
    result = across_seed_comparison("a", "b", {1: 0.5, 2: 0.5}, {1: 0.5, 2: 0.5})
    assert result.mean_diff == 0.0
    assert "identical" in result.note


def test_across_seed_comparison_survives_a_nonfinite_seed():
    result = across_seed_comparison(
        "a", "b", {1: 0.7, 2: float("nan"), 3: 0.8}, {1: 0.6, 2: 0.6, 3: 0.7}
    )
    assert result.mean_diff == pytest.approx(0.1)


# ------------------------------------------------------------------ the rest


def test_permutation_test_p_value_is_bounded_below_by_one_over_b_plus_one():
    values = [10.0] * 30 + [0.0] * 30
    labels = [1] * 30 + [0] * 30
    result = permutation_test_auc_above_chance(values, labels, n_permutations=200, rng=new_rng(0))
    assert result["auc"] == pytest.approx(1.0)
    assert result["p_value"] == pytest.approx(1 / 201)
    assert result["p_value"] > 0.0


def test_spearman_finds_a_monotone_relationship():
    result = spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
    assert result["rho"] == pytest.approx(1.0)
    assert result["n"] == 5


def test_spearman_ignores_nonfinite_pairs():
    result = spearman([1, 2, float("nan"), 4], [1, 2, 3, 4])
    assert result["n"] == 3


def test_spearman_reports_nan_for_a_constant_input():
    result = spearman([1, 1, 1, 1], [1, 2, 3, 4])
    assert math.isnan(result["rho"])


def test_summarise_across_seeds_reports_mean_sd_and_the_per_seed_values():
    summary = summarise_across_seeds({42: 0.7, 43: 0.8, 44: 0.9})
    assert summary["mean"] == pytest.approx(0.8)
    assert summary["sd"] == pytest.approx(0.1)
    assert summary["n_seeds"] == 3
    assert summary["per_seed"] == {42: 0.7, 43: 0.8, 44: 0.9}


def test_summarise_across_seeds_with_one_seed_has_no_sd():
    summary = summarise_across_seeds({42: 0.7})
    assert math.isnan(summary["sd"])
