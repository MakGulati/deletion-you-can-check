"""The sweep report, on synthetic tables whose answers are known in advance.

Every fixture here is constructed so the correct output can be written down
without running the code: a population whose losses are fixed has a perplexity
that can be computed by hand, and a score that perfectly separates two
populations has AUC 1.0 exactly.
"""

from __future__ import annotations

import json
import math
import os

import pytest

from deletion_signals.orientations import OrientationTable
from deletion_signals.schema import ScoreRow, make_flags, write_score_file
from deletion_signals.sweep_report import (
    DELETION_MIN_AUC,
    MEMBERSHIP_GONE_BAND,
    UTILITY_MAX_PPL_RATIO,
    apply_decision_rule,
    build_results,
    evaluate_cell,
    format_report,
    jsonable,
    perplexity,
    run_strength,
    token_weighted_mean_nll,
    write_plot,
)
from deletion_signals.schema import load_any


@pytest.fixture(scope="module")
def table() -> OrientationTable:
    return OrientationTable.load()


# --------------------------------------------------------- token weighting


def test_the_perplexity_aggregate_is_token_weighted():
    """A six-token sample must not weigh the same as a five-hundred-token one:
    the token-weighted mean is the only one that is a corpus perplexity."""
    mean, weighted = token_weighted_mean_nll([1.0, 3.0], [100, 300])
    assert weighted is True
    # (1*100 + 3*300) / 400 = 2.5, not the unweighted 2.0.
    assert mean == pytest.approx(2.5)
    assert perplexity(mean) == pytest.approx(math.exp(2.5))


def test_missing_token_counts_fall_back_and_say_so():
    mean, weighted = token_weighted_mean_nll([1.0, 3.0], [None, None])
    assert weighted is False
    assert mean == pytest.approx(2.0)


def test_a_population_with_no_losses_yields_none():
    mean, weighted = token_weighted_mean_nll([None, None], [10, 10])
    assert mean is None and weighted is False


def test_non_finite_losses_are_dropped_not_propagated():
    mean, _ = token_weighted_mean_nll([1.0, float("nan")], [10, 10])
    assert mean == pytest.approx(1.0)


# ------------------------------------------------------------- the synthetic run


def _rows(
    seed: int,
    *,
    forget_loss: float,
    retain_loss: float,
    test_loss: float,
    finetuned_loss: float,
    base_loss: float,
    cosine_by_population,
    n_per_population: int = 20,
):
    """One score table. Every sample in a population shares its numbers, so the
    resulting perplexities and ratios are exactly the values passed in."""
    rows = []
    losses = {"forget": forget_loss, "retain": retain_loss, "test": test_loss}
    for population, loss in losses.items():
        for index in range(n_per_population):
            # A tiny per-sample jitter, deterministic and symmetric, so the
            # AUC is driven by the population means and not by ties.
            jitter = (index - n_per_population / 2) * 1e-6
            rows.append(
                ScoreRow(
                    seed=seed,
                    method="ga_gd",
                    dataset="synthetic",
                    model_id="synthetic",
                    sample_id=f"{population}:{index}",
                    text_sha1=f"{population}{index:04d}",
                    population=population,
                    n_tokens=100,
                    loss_unlearned=loss + jitter,
                    loss_base=base_loss,
                    loss_ratio=(loss + jitter) / base_loss,
                    loss_delta=loss + jitter - base_loss,
                    loss_finetuned=finetuned_loss,
                    loss_ratio_vs_finetuned=(loss + jitter) / finetuned_loss,
                    loss_delta_vs_finetuned=loss + jitter - finetuned_loss,
                    grad_norm_sq=1.0,
                    raw_cosine=cosine_by_population[population] + jitter,
                    alignment_cos_sq=(cosine_by_population[population] + jitter) ** 2,
                    flags=make_flags(),
                )
            )
    return rows


def _write_cell(tmp_path, seed: int, ascent_lr: float, **kwargs) -> str:
    path = os.path.join(tmp_path, f"lr{ascent_lr:g}_seed{seed}.json")
    write_score_file(
        path,
        _rows(seed, **kwargs),
        {
            "seed": seed,
            "method": "ga_gd",
            "unlearning_run": {
                "hyperparameters": {"ascent_lr": ascent_lr, "descent_lr": 2e-5},
                "steps_taken": 100,
                "stop_reason": "completed",
                "final_forget_loss": kwargs["forget_loss"],
                "final_retain_loss": kwargs["retain_loss"],
            },
        },
    )
    return path


PERFECT = {"forget": 0.9, "retain": -0.9, "test": 0.0}
CHANCE = {"forget": 0.0, "retain": 0.0, "test": 0.0}


def _qualifying(tmp_path, seed: int, ascent_lr: float = 5e-5) -> str:
    """A cell built to satisfy all three criteria exactly.

    forget and test share a loss, so the membership AUC is 0.5; retain and test
    perplexities are unchanged from the fine-tuned model, so both utility ratios
    are 1.0; the cosine separates forget from retain perfectly, so the deletion
    AUC is 1.0.
    """
    return _write_cell(
        tmp_path,
        seed,
        ascent_lr,
        forget_loss=2.0,
        retain_loss=2.0,
        test_loss=2.0,
        finetuned_loss=2.0,
        base_loss=3.0,
        cosine_by_population=PERFECT,
    )


def test_a_cell_reports_the_perplexities_it_was_built_from(tmp_path, table):
    path = _qualifying(tmp_path, 42)
    cell = evaluate_cell(load_any(path), table, n_bootstrap=200, confidence=0.95)

    assert cell.seed == 42
    assert cell.strength == pytest.approx(5e-5)
    assert cell.perplexities["retain"]["unlearned"] == pytest.approx(math.exp(2.0), rel=1e-4)
    assert cell.perplexities["retain"]["ratio"] == pytest.approx(1.0, rel=1e-4)
    assert cell.forget_retain_loss_ratio == pytest.approx(1.0, rel=1e-4)
    assert cell.token_weighted is True


def test_a_perfectly_separating_cosine_scores_one(tmp_path, table):
    cell = evaluate_cell(
        load_any(_qualifying(tmp_path, 42)), table, n_bootstrap=200, confidence=0.95
    )
    assert cell.auc("forget_vs_retain", "raw_cosine") == pytest.approx(1.0)
    assert cell.auc("forget_vs_unseen", "loss_ratio") == pytest.approx(0.5, abs=0.02)


def test_the_strength_comes_from_the_recorded_hyperparameters_not_the_path():
    metadata = {"unlearning_run": {"hyperparameters": {"ascent_lr": 2e-4}}}
    strength, source = run_strength(metadata, "ga_gd")
    assert strength == pytest.approx(2e-4)
    assert "unlearning_run" in source


def test_the_strength_falls_back_to_the_effective_config():
    metadata = {
        "provenance": {
            "effective_config": {"unlearning": {"ga_gd": {"ascent_lr": 1e-5}}}
        }
    }
    strength, source = run_strength(metadata, "ga_gd")
    assert strength == pytest.approx(1e-5)
    assert "effective_config" in source


def test_a_run_with_no_recorded_strength_is_rejected():
    """A sweep point with no strength is unplaceable; guessing one from the
    directory name would turn a mislabelled cell into a monotone curve."""
    with pytest.raises(ValueError, match="unplaceable"):
        run_strength({}, "ga_gd")


# ------------------------------------------------------------- the decision rule


def test_the_qualifying_cell_meets_all_three_criteria(tmp_path, table):
    cell = evaluate_cell(
        load_any(_qualifying(tmp_path, 42)), table, n_bootstrap=500, confidence=0.95
    )
    criteria = cell.criteria
    assert criteria["membership_gone"]["met"] is True
    assert criteria["utility_preserved"]["met"] is True
    assert criteria["deletion_present"]["met"] is True
    assert criteria["qualifies"] is True


def test_surviving_membership_evidence_fails_the_first_criterion(tmp_path, table):
    """forget loss well above test loss: loss_ratio still separates them, so the
    membership evidence has not gone anywhere."""
    path = _write_cell(
        tmp_path, 42, 1e-5,
        forget_loss=6.0, retain_loss=2.0, test_loss=2.0,
        finetuned_loss=2.0, base_loss=3.0, cosine_by_population=PERFECT,
    )
    cell = evaluate_cell(load_any(path), table, n_bootstrap=200, confidence=0.95)
    assert cell.criteria["membership_gone"]["met"] is False
    assert cell.criteria["membership_gone"]["auc"] > MEMBERSHIP_GONE_BAND[1]
    assert cell.criteria["qualifies"] is False


def test_collapsed_utility_fails_the_second_criterion(tmp_path, table):
    path = _write_cell(
        tmp_path, 42, 2e-4,
        forget_loss=2.0, retain_loss=8.0, test_loss=8.0,
        finetuned_loss=2.0, base_loss=3.0, cosine_by_population=PERFECT,
    )
    cell = evaluate_cell(load_any(path), table, n_bootstrap=200, confidence=0.95)
    assert cell.criteria["utility_preserved"]["met"] is False
    assert cell.criteria["utility_preserved"]["ppl_ratio"]["retain"] > UTILITY_MAX_PPL_RATIO


def test_an_absent_finetuned_loss_makes_utility_unevaluable_not_passing(tmp_path, table):
    """A table with no loss_finetuned cannot show utility was preserved. The
    criterion must fail, not pass by default on missing evidence."""
    rows = _rows(
        42, forget_loss=2.0, retain_loss=2.0, test_loss=2.0,
        finetuned_loss=2.0, base_loss=3.0, cosine_by_population=PERFECT,
    )
    for row in rows:
        row.loss_finetuned = None
        row.loss_ratio_vs_finetuned = None
        row.loss_delta_vs_finetuned = None
    path = os.path.join(tmp_path, "no_finetuned.json")
    write_score_file(path, rows, {
        "seed": 42, "method": "ga_gd",
        "unlearning_run": {"hyperparameters": {"ascent_lr": 1e-5}},
    })

    cell = evaluate_cell(load_any(path), table, n_bootstrap=200, confidence=0.95)
    assert cell.criteria["utility_preserved"]["met"] is False
    assert cell.criteria["utility_preserved"]["evaluable"] is False


def test_a_chance_level_cosine_fails_the_deletion_criterion(tmp_path, table):
    path = _write_cell(
        tmp_path, 42, 1e-4,
        forget_loss=2.0, retain_loss=2.0, test_loss=2.0,
        finetuned_loss=2.0, base_loss=3.0, cosine_by_population=CHANCE,
    )
    cell = evaluate_cell(load_any(path), table, n_bootstrap=200, confidence=0.95)
    assert cell.criteria["deletion_present"]["met"] is False
    assert cell.criteria["deletion_present"]["auc"] < DELETION_MIN_AUC


def test_the_rule_needs_the_ci_as_well_as_the_point_estimate():
    """A point estimate over the threshold with an interval straddling chance
    does not count: the rule requires both."""
    from deletion_signals.sweep_report import CellRun

    cell = CellRun(path="", seed=42, method="ga_gd", strength=1e-5, strength_source="test")
    cell.aucs = {
        "forget_vs_retain": {"raw_cosine": {"auc": 0.80, "ci_lower": 0.42}},
        "forget_vs_unseen": {"loss_ratio": {"auc": 0.50}},
    }
    cell.perplexities = {p: {"ratio": 1.0} for p in ("forget", "retain", "test")}
    criteria = apply_decision_rule(cell)
    assert criteria["deletion_present"]["met"] is False
    assert criteria["membership_gone"]["met"] is True


# ------------------------------------------------------------- across settings


def _three_seed_sweep(tmp_path):
    paths = []
    for seed in (42, 43, 44):
        paths.append(_qualifying(tmp_path, seed, ascent_lr=5e-5))
        paths.append(
            _write_cell(
                tmp_path, seed, 1e-5,
                forget_loss=4.0, retain_loss=2.0, test_loss=2.0,
                finetuned_loss=2.0, base_loss=3.0, cosine_by_population=PERFECT,
            )
        )
    return paths


def test_settings_are_grouped_by_strength_with_seed_as_the_unit(tmp_path, table):
    results = build_results(
        [load_any(p) for p in _three_seed_sweep(tmp_path)], table, 300, 0.95
    )
    assert [s["strength"] for s in results["settings"]] == [1e-5, 5e-5]
    for setting in results["settings"]:
        assert setting["n_seeds"] == 3
        assert setting["seeds"] == [42, 43, 44]


def test_a_setting_qualifies_only_when_every_seed_does(tmp_path, table):
    paths = _three_seed_sweep(tmp_path)
    results = build_results([load_any(p) for p in paths], table, 300, 0.95)
    strong = next(s for s in results["settings"] if s["strength"] == 5e-5)
    weak = next(s for s in results["settings"] if s["strength"] == 1e-5)

    assert strong["criteria"]["qualifies_on_all_seeds"] is True
    # The weak cell keeps its membership signal, so it fails on every seed.
    assert weak["criteria"]["n_seeds_meeting"]["membership_gone"] == 0
    assert weak["criteria"]["qualifies_on_all_seeds"] is False
    assert results["verdict"]["qualifying_settings"] == [
        {"method": "ga_gd", "strength": 5e-5}
    ]
    assert "fingerprint persists" in results["verdict"]["outcome"]


def test_one_failing_seed_disqualifies_the_setting(tmp_path, table):
    paths = [_qualifying(tmp_path, seed, 5e-5) for seed in (42, 43)]
    paths.append(
        _write_cell(
            tmp_path, 44, 5e-5,
            forget_loss=2.0, retain_loss=2.0, test_loss=2.0,
            finetuned_loss=2.0, base_loss=3.0, cosine_by_population=CHANCE,
        )
    )
    results = build_results([load_any(p) for p in paths], table, 300, 0.95)
    setting = results["settings"][0]
    assert setting["criteria"]["n_seeds_meeting"]["deletion_present"] == 2
    assert setting["criteria"]["qualifies_on_all_seeds"] is False
    assert results["verdict"]["qualifying_settings"] == []


def test_two_files_for_the_same_strength_and_seed_are_rejected(tmp_path, table):
    """One would silently overwrite the other in the across-seed summary."""
    first = _qualifying(tmp_path, 42, 5e-5)
    second = os.path.join(tmp_path, "duplicate.json")
    write_score_file(
        second,
        _rows(42, forget_loss=2.0, retain_loss=2.0, test_loss=2.0,
              finetuned_loss=2.0, base_loss=3.0, cosine_by_population=PERFECT),
        {"seed": 42, "method": "ga_gd",
         "unlearning_run": {"hyperparameters": {"ascent_lr": 5e-5}}},
    )
    with pytest.raises(ValueError, match="share a"):
        build_results([load_any(first), load_any(second)], table, 100, 0.95)


def test_paired_differences_are_computed_within_seed(tmp_path, table):
    """raw_cosine separates perfectly and loss_ratio is at chance here, so the
    paired difference is 0.5 on every seed and its across-seed SD is 0."""
    results = build_results(
        [load_any(_qualifying(tmp_path, seed, 5e-5)) for seed in (42, 43, 44)],
        table, 300, 0.95,
    )
    setting = results["settings"][0]
    against_loss = next(
        c for c in setting["paired_differences"] if c["score_b"] == "loss_ratio"
    )
    assert against_loss["mean"] == pytest.approx(0.5, abs=0.02)
    assert against_loss["n_seeds"] == 3


def test_across_seed_sd_is_reported_for_every_headline_quantity(tmp_path, table):
    results = build_results(
        [load_any(_qualifying(tmp_path, seed, 5e-5)) for seed in (42, 43, 44)],
        table, 300, 0.95,
    )
    setting = results["settings"][0]
    for summary in (
        setting["perplexity_ratio"]["retain"],
        setting["forget_retain_loss_ratio"],
        setting["auc"]["forget_vs_retain"]["raw_cosine"],
    ):
        assert summary["n_seeds"] == 3
        assert summary["sd"] is not None and math.isfinite(summary["sd"])


# ------------------------------------------------------------------- outcomes


def test_the_incomplete_forgetting_outcome_is_named_as_such(tmp_path, table):
    """Membership gone and utility intact, but no deletion signal: the baseline
    fingerprint was consistent with unlearning not having finished."""
    paths = [
        _write_cell(
            tmp_path, seed, 1e-4,
            forget_loss=2.0, retain_loss=2.0, test_loss=2.0,
            finetuned_loss=2.0, base_loss=3.0, cosine_by_population=CHANCE,
        )
        for seed in (42, 43, 44)
    ]
    results = build_results([load_any(p) for p in paths], table, 300, 0.95)
    assert "incomplete forgetting" in results["verdict"]["outcome"]


def test_the_inconclusive_outcome_is_named_as_such(tmp_path, table):
    """Utility collapsed everywhere, so the sweep never reached a state in which
    the question could be asked."""
    paths = [
        _write_cell(
            tmp_path, seed, 2e-4,
            forget_loss=9.0, retain_loss=9.0, test_loss=9.0,
            finetuned_loss=2.0, base_loss=3.0, cosine_by_population=PERFECT,
        )
        for seed in (42, 43, 44)
    ]
    results = build_results([load_any(p) for p in paths], table, 300, 0.95)
    assert "inconclusive" in results["verdict"]["outcome"]


# -------------------------------------------------------------------- outputs


def test_the_report_states_the_thresholds_and_the_verdict(tmp_path, table):
    results = build_results(
        [load_any(p) for p in _three_seed_sweep(tmp_path)], table, 300, 0.95
    )
    report = format_report(results)

    assert "PREREGISTRATION.md" in report
    assert str(MEMBERSHIP_GONE_BAND[0]) in report and str(DELETION_MIN_AUC) in report
    assert "white-box deletion audit" in report
    assert "promoted to the headline after being seen to win" in report
    assert "n = 3 seeds" in report
    assert results["verdict"]["outcome"].split(":")[0] in report


def test_the_results_json_is_strict(tmp_path, table):
    results = build_results(
        [load_any(p) for p in _three_seed_sweep(tmp_path)], table, 200, 0.95
    )
    encoded = json.dumps(jsonable(results), allow_nan=False)
    assert "NaN" not in encoded and "Infinity" not in encoded


def test_the_figure_is_written_once_there_are_two_strengths(tmp_path, table):
    results = build_results(
        [load_any(p) for p in _three_seed_sweep(tmp_path)], table, 200, 0.95
    )
    path = write_plot(results, os.path.join(tmp_path, "sweep.png"))
    assert path is not None and os.path.getsize(path) > 0


def test_a_single_strength_writes_no_figure(tmp_path, table):
    """One point is not a sweep, and a line through it would imply a trend."""
    results = build_results(
        [load_any(_qualifying(tmp_path, seed, 5e-5)) for seed in (42, 43, 44)],
        table, 200, 0.95,
    )
    assert write_plot(results, os.path.join(tmp_path, "sweep.png")) is None
