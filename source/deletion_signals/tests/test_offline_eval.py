"""Offline recomputation from a synthetic table whose answers are known."""

from __future__ import annotations

import json
import math
import os

import pytest

from deletion_signals.config import load_config
from deletion_signals.offline_eval import (
    PRIMARY_COMPARISON,
    evaluate,
    expand_inputs,
    format_report,
    main,
)
from deletion_signals.orientations import OrientationTable
from deletion_signals.schema import ScoreRow, make_flags, text_fingerprint, write_score_file


def _rows(seed: int, method: str, separation: float, n: int = 30):
    """Forget samples score `separation` higher than retain and test samples.

    Perfectly separable when separation > 0, so every oriented AUC is 1.0 and
    the recomputed numbers can be checked against an exact expectation.
    """
    rows = []
    for population, offset in (("forget", separation), ("retain", 0.0), ("test", -0.5)):
        for index in range(n):
            jitter = index / (10.0 * n)
            rows.append(
                ScoreRow(
                    seed=seed,
                    method=method,
                    dataset="synthetic",
                    model_id="synthetic-model",
                    sample_id=f"{population}:{index}",
                    text_sha1=text_fingerprint(f"{population}-{index}"),
                    population=population,
                    n_tokens=32,
                    # higher_is_positive on forget_vs_retain
                    alignment_cos_sq=0.001 + offset + jitter,
                    grad_norm_sq=10.0 + offset + jitter,
                    # lower_is_positive on member_vs_unseen: members score lower
                    loss_unlearned=2.0 + offset + jitter,
                    loss_base=2.0,
                    loss_ratio=(2.0 + offset + jitter) / 2.0,
                    loss_delta=offset + jitter,
                    grad_norm_sq_ref=10.0,
                    grad_ratio=(10.0 + offset + jitter) / 10.0,
                    log_grad_ratio=math.log(10.0 + offset + jitter) - math.log(10.0),
                    fuma_lora_ratio=1.0 + offset + jitter,
                    raw_cosine=0.03,
                    sif=100.0 + offset + jitter,
                    combined_log_score=offset + jitter,
                    flags=make_flags(),
                )
            )
    return rows


def _write(tmp_path, seed: int, method: str, separation: float = 1.0) -> str:
    directory = os.path.join(tmp_path, f"seed{seed}", method)
    path = os.path.join(directory, "per_sample_scores.json")
    write_score_file(path, _rows(seed, method, separation), {"seed": seed, "method": method})
    return path


@pytest.fixture(scope="module")
def table() -> OrientationTable:
    return OrientationTable.load()


@pytest.fixture()
def config():
    cfg = load_config()
    cfg["evaluation"]["n_bootstrap"] = 50
    cfg["evaluation"]["fpr_targets"] = [0.001, 0.05]
    return cfg


def _load(paths):
    from deletion_signals.schema import load_any

    return [load_any(p) for p in paths]


# -------------------------------------------------------------- recomputation


def test_perfect_separation_recovers_auc_one(tmp_path, config, table):
    files = _load([_write(tmp_path, 42, "ga_gd")])
    results = evaluate(files, config, table)
    entry = results["per_run"]["ga_gd/seed42"]["tasks"]["forget_vs_retain"]["scores"][
        "alignment_cos_sq"
    ]
    assert entry["auc"] == pytest.approx(1.0)
    assert entry["average_precision"] == pytest.approx(1.0)
    assert entry["n_positive"] == 30 and entry["n_negative"] == 30


def test_an_inverted_score_is_reported_below_half_not_flipped(tmp_path, config, table):
    """loss_unlearned is declared lower_is_positive on member_vs_unseen, but in
    this synthetic table members score HIGHER. The AUC must come out below 0.5."""
    files = _load([_write(tmp_path, 42, "ga_gd")])
    results = evaluate(files, config, table)
    entry = results["per_run"]["ga_gd/seed42"]["tasks"]["member_vs_unseen"]["scores"][
        "loss_unlearned"
    ]
    assert entry["direction"] == "lower_is_positive"
    assert entry["auc"] < 0.5


def test_bootstrap_interval_is_attached_to_every_available_score(tmp_path, config, table):
    files = _load([_write(tmp_path, 42, "ga_gd")])
    results = evaluate(files, config, table)
    for entry in results["per_run"]["ga_gd/seed42"]["tasks"]["forget_vs_retain"]["scores"].values():
        if entry.get("available"):
            assert entry["ci_lower"] <= entry["auc"] <= entry["ci_upper"]
            assert entry["n_bootstrap"] > 0


def test_a_tenth_of_a_percent_fpr_is_suppressed_at_this_sample_size(tmp_path, config, table):
    files = _load([_write(tmp_path, 42, "ga_gd")])
    results = evaluate(files, config, table)
    entry = results["per_run"]["ga_gd/seed42"]["tasks"]["forget_vs_retain"]["scores"][
        "alignment_cos_sq"
    ]
    assert entry["tpr_at_fpr"]["0.001"]["resolvable"] is False
    assert entry["tpr_at_fpr"]["0.001"]["tpr"] is None
    assert entry["tpr_at_fpr"]["0.05"]["resolvable"] is True


# ------------------------------------------------------------- across seeds


def test_across_seed_summary_covers_every_seed(tmp_path, config, table):
    paths = [_write(tmp_path, seed, "ga_gd", 1.0) for seed in (42, 43, 44)]
    results = evaluate(_load(paths), config, table)
    summary = results["across_seeds"]["ga_gd"]["forget_vs_retain"]["alignment_cos_sq"]
    assert summary["n_seeds"] == 3
    assert set(summary["per_seed"]) == {42, 43, 44}
    assert summary["mean"] == pytest.approx(1.0)


def test_paired_across_seeds_uses_the_seed_as_the_unit(tmp_path, config, table):
    paths = [_write(tmp_path, seed, "ga_gd") for seed in (42, 43, 44)]
    results = evaluate(_load(paths), config, table)
    comparisons = results["paired_across_seeds"]["ga_gd"]["forget_vs_retain"]
    assert comparisons
    for comparison in comparisons:
        assert comparison["unit"].startswith("seed")
        assert comparison["n_seeds"] == 3
        # Everything is compared against the head of PRIMARY_COMPARISON, which
        # is the preregistered primary score.
        assert comparison["score_a"] == PRIMARY_COMPARISON[0]


def test_the_primary_comparison_is_headed_by_the_preregistered_primary_score():
    """The signed cosine, not its square: the difference between the two is
    exactly the sign information, so a comparison headed by cos^2 could not
    measure what squaring costs."""
    assert PRIMARY_COMPARISON[0] == "raw_cosine"
    assert PRIMARY_COMPARISON[1] == "alignment_cos_sq"
    # A baseline, never presented as primary.
    assert PRIMARY_COMPARISON[-1] == "combined_log_score"


def test_two_methods_are_kept_apart(tmp_path, config, table):
    paths = [
        _write(tmp_path, 42, "ga_gd"),
        _write(tmp_path, 42, "scrub"),
    ]
    results = evaluate(_load(paths), config, table)
    assert set(results["across_seeds"]) == {"ga_gd", "scrub"}
    assert "ga_gd/seed42" in results["per_run"]
    assert "scrub/seed42" in results["per_run"]


# ------------------------------------------------------------------- legacy


LEGACY = {
    population: [
        {
            "score": 1.0,
            "log_score": value,
            "sif_value": 100.0 + value,
            "grad_norm_sq": 10.0 + value,
            "cos_sq": 0.001 + value,
            "fisher_explained": 0.04,
        }
        for value in values
    ]
    for population, values in (
        ("forget", [1.0, 1.1, 1.2, 1.3]),
        ("retain", [0.0, 0.1, 0.2, 0.3]),
        ("test", [-0.5, -0.4, -0.3, -0.2]),
    )
}


def _write_legacy(tmp_path, seed: int, method: str) -> str:
    directory = os.path.join(tmp_path, f"results_seed{seed}", method)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "per_sample_components.json")
    with open(path, "w") as handle:
        json.dump(LEGACY, handle)
    return path


def test_legacy_inputs_evaluate_with_only_the_scores_they_carry(tmp_path, config, table):
    paths = [_write_legacy(tmp_path, seed, "ga_gd") for seed in (42, 43, 44)]
    results = evaluate(_load(paths), config, table)

    assert set(results["metadata"]["scores_evaluated"]) == {
        "grad_norm_sq", "alignment_cos_sq", "sif", "combined_log_score"
    }
    for absent in ("loss_ratio", "fuma_lora_ratio", "grad_ratio"):
        assert absent in results["metadata"]["scores_absent_from_inputs"]

    entry = results["per_run"]["ga_gd/seed42"]["tasks"]["forget_vs_retain"]["scores"][
        "alignment_cos_sq"
    ]
    assert entry["auc"] == pytest.approx(1.0)


def test_legacy_and_modern_inputs_can_be_evaluated_together(tmp_path, config, table):
    paths = [_write_legacy(tmp_path, 42, "ga_gd"), _write(tmp_path, 43, "ga_gd")]
    results = evaluate(_load(paths), config, table)
    assert len(results["metadata"]["legacy_inputs"]) == 1
    assert set(results["across_seeds"]["ga_gd"]["forget_vs_retain"]["alignment_cos_sq"]["per_seed"]) == {42, 43}


# -------------------------------------------------------------------- report


def test_report_renders_and_states_the_orientation_policy(tmp_path, config, table):
    files = _load([_write(tmp_path, 42, "ga_gd")])
    report = format_report(evaluate(files, config, table), table)
    assert "max(AUC, 1-AUC)` is not used" in report
    assert "forget_vs_retain" in report
    assert "alignment_cos_sq" in report


def test_report_lists_scores_absent_from_the_inputs(tmp_path, config, table):
    paths = [_write_legacy(tmp_path, 42, "ga_gd")]
    report = format_report(evaluate(_load(paths), config, table), table)
    assert "Scores absent from the inputs" in report
    assert "fuma_lora_ratio" in report


# ----------------------------------------------------------------------- CLI


def test_cli_writes_results_and_report(tmp_path, monkeypatch):
    _write(tmp_path, 42, "ga_gd")
    _write(tmp_path, 43, "ga_gd")
    output = os.path.join(tmp_path, "evaluation")

    exit_code = main([
        "--inputs", os.path.join(tmp_path, "seed*", "*", "per_sample_scores.json"),
        "--output", output,
        "--n-bootstrap", "20",
        "--quiet",
    ])
    assert exit_code == 0
    assert os.path.exists(os.path.join(output, "results.json"))
    assert os.path.exists(os.path.join(output, "report.md"))

    with open(os.path.join(output, "results.json")) as handle:
        results = json.load(handle)
    assert set(results["per_run"]) == {"ga_gd/seed42", "ga_gd/seed43"}


def test_cli_reads_legacy_files(tmp_path):
    for seed in (42, 43):
        _write_legacy(tmp_path, seed, "ga_gd")
    output = os.path.join(tmp_path, "legacy_evaluation")
    exit_code = main([
        "--legacy",
        "--inputs", os.path.join(tmp_path, "results_seed*", "*", "per_sample_components.json"),
        "--output", output,
        "--n-bootstrap", "20",
        "--quiet",
    ])
    assert exit_code == 0
    with open(os.path.join(output, "results.json")) as handle:
        results = json.load(handle)
    assert len(results["metadata"]["legacy_inputs"]) == 2


def test_cli_results_are_strict_json(tmp_path):
    """No NaN or Infinity literals: those are not valid JSON and strict parsers
    reject them, while `null` says 'not computed' in a form everyone reads."""
    _write(tmp_path, 42, "ga_gd")
    output = os.path.join(tmp_path, "evaluation")
    main(["--inputs", os.path.join(tmp_path, "seed*", "*", "per_sample_scores.json"),
          "--output", output, "--n-bootstrap", "20", "--quiet"])
    with open(os.path.join(output, "results.json")) as handle:
        text = handle.read()
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-strict literal {c}"))


def test_jsonable_replaces_nonfinite_floats_only():
    from deletion_signals.offline_eval import jsonable

    payload = {"a": float("nan"), "b": float("-inf"), "c": 1.5, "d": [float("inf"), "x", None]}
    assert jsonable(payload) == {"a": None, "b": None, "c": 1.5, "d": [None, "x", None]}


def test_expand_inputs_rejects_a_pattern_that_matches_nothing(tmp_path):
    with pytest.raises(FileNotFoundError):
        expand_inputs([os.path.join(tmp_path, "nothing", "*.json")])


def test_expand_inputs_deduplicates(tmp_path):
    path = _write(tmp_path, 42, "ga_gd")
    assert expand_inputs([path, path]) == [path]
