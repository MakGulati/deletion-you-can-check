"""The classification partition, the pass counts, and the output schema.

Every verdict in the paper comes out of :func:`classify_run`, so this file
enumerates the combinations rather than sampling them: four booleans and a
stopping reason, checked against the label each is required to produce. A
partition that overlapped or left a gap would put a run in the wrong row of
Table 2 without anything failing.
"""

from __future__ import annotations

import itertools
import json
import math

import pytest

from deletion_signals.fllm_report import (
    CLASSIFICATION_ORDER,
    DELETION_NO_UTILITY,
    FAILED_RUN,
    GLOBAL_DESTRUCTION,
    INCOMPLETE,
    INCONCLUSIVE,
    SUCCESSFUL,
    build_document,
    classify_audit,
    classify_run,
    format_report,
    summarise_setting,
    table4_npo_sweep,
)


def _run(
    membership=True,
    selective=True,
    utility=True,
    stop_reason="completed",
    seed=42,
    method="npo",
    setting="lr=2.8e-05, beta=0.1",
    membership_auc=0.5,
    utility_evaluable=True,
    selective_evaluable=True,
):
    return {
        "seed": seed,
        "method": method,
        "setting": setting,
        "path": f"/tmp/{method}-{seed}.json",
        "perplexity_ratio": {"retain": 1.02, "test": 1.01},
        "rnff": {"forget": 0.8, "retain": 0.1},
        "data_quality": {"truncated": 0, "nonfinite_counts": {},
                         "populations_disjoint": True, "null_counts": {}},
        "unlearning": {"stop_reason": stop_reason, "steps_taken": 100,
                       "steps_configured": 100},
        "axes": {
            "membership_removed": {"met": membership, "auc": membership_auc},
            "selective_forgetting": {"met": selective, "evaluable": selective_evaluable},
            "utility_preserved": {"met": utility, "evaluable": utility_evaluable},
            "deletion_detectable": {"auc": 0.9},
        },
    }


# ------------------------------------------------------------ the partition


def test_the_partition_is_total_and_deterministic():
    """Every combination of the four criteria yields exactly one known label."""
    for membership, selective, utility, completed in itertools.product(
        [True, False], repeat=4
    ):
        verdict = classify_run(
            _run(
                membership=membership,
                selective=selective,
                utility=utility,
                stop_reason="completed" if completed else "collapse_guard",
            )
        )
        assert verdict.classification in CLASSIFICATION_ORDER
        assert verdict.conjunction == (membership and selective and utility and completed)


def test_all_criteria_met_is_successful_selective_deletion():
    verdict = classify_run(_run())
    assert verdict.classification == SUCCESSFUL
    assert verdict.conjunction is True


def test_a_collapse_guard_trip_is_a_failed_run_whatever_the_axes_say():
    """A destroyed model that happens to land in the chance region is not a
    success; the run did not complete, and that takes precedence."""
    verdict = classify_run(_run(stop_reason="collapse_guard"))
    assert verdict.classification == FAILED_RUN
    assert verdict.conjunction is False
    assert verdict.collapse_guard_not_triggered is False


def test_a_non_finite_loss_is_a_failed_run():
    assert classify_run(_run(stop_reason="non_finite_loss")).classification == FAILED_RUN


def test_selective_forgetting_without_utility_is_its_own_label():
    verdict = classify_run(_run(selective=True, utility=False))
    assert verdict.classification == DELETION_NO_UTILITY


def test_broken_utility_with_unspecific_damage_is_global_destruction():
    verdict = classify_run(_run(membership=True, selective=False, utility=False))
    assert verdict.classification == GLOBAL_DESTRUCTION


def test_the_global_destruction_reason_does_not_overclaim_magnitude():
    """The label fires on a criterion pattern; an 18% perplexity rise triggers
    it, so the stated reason must not describe total destruction."""
    reason = classify_run(_run(selective=False, utility=False)).reason
    assert "specificity budget" in reason
    assert "nearly as far" not in reason


def test_a_useful_model_that_still_leaks_membership_is_incomplete_forgetting():
    verdict = classify_run(_run(membership=False, selective=False, utility=True))
    assert verdict.classification == INCOMPLETE


def test_membership_removed_without_selectivity_is_not_called_a_success():
    """Membership AUC near chance never certifies anything on its own."""
    verdict = classify_run(_run(membership=True, selective=False, utility=True))
    assert verdict.classification != SUCCESSFUL
    assert verdict.classification == INCONCLUSIVE


def test_an_unmeasurable_criterion_is_not_a_satisfied_one():
    """The failure mode this guards: a missing RNFF denominator would leave
    selectivity unevaluated, and treating that as met is how a destroyed model
    slips through."""
    verdict = classify_run(_run(selective_evaluable=False))
    assert verdict.classification == INCONCLUSIVE
    assert "selective_forgetting" in verdict.unevaluable
    assert "not a satisfied one" in verdict.reason


def test_an_absent_membership_auc_is_recorded_as_unevaluable():
    verdict = classify_run(_run(membership_auc=None))
    assert "membership_removed" in verdict.unevaluable


# --------------------------------------------------------------- pass counts


def test_marginal_and_conjunction_counts_are_reported_separately():
    """Three seeds each passing a different criterion: every marginal count is
    1, the conjunction count is 0. Reporting only the marginals would read as
    partial success."""
    verdicts = [
        classify_run(_run(seed=42, membership=True, selective=False, utility=False)),
        classify_run(_run(seed=43, membership=False, selective=True, utility=False)),
        classify_run(_run(seed=44, membership=False, selective=False, utility=True)),
    ]
    block = summarise_setting(verdicts)

    assert block["marginal_pass_counts"]["membership_removed"] == 1
    assert block["marginal_pass_counts"]["selective_forgetting"] == 1
    assert block["marginal_pass_counts"]["utility_preserved"] == 1
    assert block["conjunction_pass_count"] == 0


def test_a_setting_whose_seeds_disagree_is_not_classified():
    verdicts = [
        classify_run(_run(seed=42)),                       # successful
        classify_run(_run(seed=43, membership=False)),     # incomplete
        classify_run(_run(seed=44)),                       # successful
    ]
    block = summarise_setting(verdicts)

    assert block["seeds_agree"] is False
    assert block["classification"] == INCONCLUSIVE
    assert "disagree" in block["classification_note"]
    assert block["per_seed_classification"] == {
        42: SUCCESSFUL, 43: INCOMPLETE, 44: SUCCESSFUL
    }
    # ...but the seed that did pass everything is still counted.
    assert block["conjunction_pass_count"] == 2


def test_unanimous_seeds_take_the_shared_label():
    verdicts = [classify_run(_run(seed=s)) for s in (42, 43, 44)]
    block = summarise_setting(verdicts)
    assert block["seeds_agree"] is True
    assert block["classification"] == SUCCESSFUL
    assert block["conjunction_pass_count"] == 3


def test_collapsed_runs_are_counted_and_not_dropped():
    verdicts = [
        classify_run(_run(seed=42, stop_reason="collapse_guard")),
        classify_run(_run(seed=43, stop_reason="collapse_guard")),
        classify_run(_run(seed=44)),
    ]
    block = summarise_setting(verdicts)

    assert block["collapse_count"] == 2
    assert block["failed_run_count"] == 2
    assert block["n_seeds"] == 3          # still three rows, nothing filtered


def test_settings_are_grouped_without_pooling_different_hyperparameters():
    audit = {
        "runs": [
            _run(seed=42, setting="lr=1e-05, beta=0.1"),
            _run(seed=43, setting="lr=1e-05, beta=0.1"),
            _run(seed=42, setting="lr=0.0001, beta=0.1"),
        ]
    }
    classified = classify_audit(audit)
    settings = {block["setting"] for block in classified["settings"]}
    assert settings == {"lr=1e-05, beta=0.1", "lr=0.0001, beta=0.1"}


# ------------------------------------------------------------ tables and output


def _audit():
    return {
        "metadata": {
            "preregistration": "configs/preregistration_v3_fllm.yaml",
            "primary_detector": "loss_ratio_vs_finetuned",
            "primary_task": "forget_vs_retain",
            "retrain_control_seeds": [42, 43, 44],
            "null_refit_floors": {},
            "thresholds": {"max_perplexity_ratio": 1.1},
        },
        "runs": [
            _run(seed=42, method="npo", setting="lr=1e-05, beta=0.1",
                 membership=False, selective=False),
            _run(seed=43, method="npo", setting="lr=1e-05, beta=0.1",
                 membership=False, selective=False),
            _run(seed=42, method="npo", setting="lr=0.0001, beta=0.1",
                 stop_reason="collapse_guard"),
            _run(seed=42, method="exact_retrain", setting="exact_retrain"),
        ],
        "settings": [],
    }


def test_table4_lists_every_strength_and_every_seed_including_failures():
    audit = _audit()
    rows = table4_npo_sweep(audit, classify_audit(audit))

    assert len(rows) == 3                      # every NPO run, none dropped
    collapsed = [r for r in rows if r["stop_reason"] == "collapse_guard"]
    assert len(collapsed) == 1
    assert collapsed[0]["classification"] == FAILED_RUN


def test_table4_excludes_non_npo_runs():
    audit = _audit()
    rows = table4_npo_sweep(audit, classify_audit(audit))
    assert all("exact_retrain" not in str(r["setting"]) for r in rows)


def test_table2_omits_the_nuisance_controls_but_keeps_exact_retraining():
    document = build_document(_audit())
    methods = {row["method"] for row in document["table2_method_verdicts"]}
    assert "npo" in methods
    assert "exact_retrain" in methods
    assert "null_refit" not in methods
    assert "routine_update" not in methods


def test_table1_lists_every_control_condition_even_when_not_run():
    document = build_document(_audit())
    conditions = [row["condition"] for row in document["table1_control_calibration"]]

    assert conditions[0] == "finetuned_reference"
    for expected in ("null_refit", "routine_update", "exact_retrain",
                     "exact_retrain_routine_update"):
        assert expected in conditions
    not_run = [r for r in document["table1_control_calibration"]
               if r.get("status") == "not run"]
    assert {r["condition"] for r in not_run} >= {"routine_update"}


def test_the_failure_log_records_every_failed_run():
    document = build_document(_audit())
    log = document["failure_log"]
    assert log["n_failed_runs"] == 1
    assert log["failed_runs"][0]["classification"] == FAILED_RUN


def test_the_document_is_strict_json_serialisable():
    """Report JSON is strict: a non-finite value must be null, meaning 'not
    computed'. allow_nan=False raises rather than emitting NaN."""
    document = build_document(_audit())
    json.dumps(document, allow_nan=False, default=str)


def test_non_finite_values_from_the_audit_become_null():
    """Regression: a single-seed across-seed SD is NaN, and a perplexity ratio
    is NaN when its denominator was unavailable. Both reach the document
    legitimately, and both would make json.dump(allow_nan=False) reject the
    whole file rather than the one field."""
    audit = _audit()
    audit["runs"][0]["perplexity_ratio"]["retain"] = float("nan")
    audit["runs"][0]["rnff"]["forget"] = float("inf")
    audit["settings"] = [{"method": "npo", "auc": {"forget_vs_retain":
                                                   {"loss_unlearned": {"mean": 0.9,
                                                                       "sd": float("nan")}}}}]

    document = build_document(audit)
    json.dumps(document, allow_nan=False, default=str)

    npo_row = next(r for r in document["table4_npo_sweep"]
                   if r["seed"] == 42 and "1e-05" in str(r["setting"]))
    assert npo_row["retain_ppl_ratio"] is None
    assert npo_row["rnff_forget"] is None


def test_the_sanitiser_leaves_finite_values_and_structure_alone():
    from deletion_signals.fllm_report import strict_json_safe

    payload = {"a": 1.5, "b": [0.0, float("nan"), {"c": float("-inf")}],
               "d": True, "e": "text", "f": None}
    assert strict_json_safe(payload) == {
        "a": 1.5, "b": [0.0, None, {"c": None}], "d": True, "e": "text", "f": None
    }


def test_the_report_renders_and_carries_its_caveats():
    text = format_report(build_document(_audit()))

    assert "Table 1" in text and "Table 2" in text
    assert "Table 3" in text and "Table 4" in text
    assert "n = 3 seeds" in text
    assert "not proof of forgetting" in text
    assert "not black-box membership-inference attacks" in text
    assert "off the label" in text          # the magnitude-vs-label warning


def test_classification_is_stable_across_repeated_builds():
    audit = _audit()
    first = build_document(audit)["classified"]
    second = build_document(audit)["classified"]
    assert first == second
