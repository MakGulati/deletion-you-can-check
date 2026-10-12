"""The FLLM audit's four tables, and the verdicts behind them.

This module computes nothing about a model. It consumes what the three
measurement components already wrote --

    fllm_audit/.../audit_results.json    the three axes, per run and seed
    fllm_audit/lowprev/lowprev_results.json
    fllm_audit/benchmark/benchmark.json

-- and turns them into Tables 1 to 4 plus a Markdown report. Keeping it purely
derivative matters: the axes have exactly one implementation, in
:mod:`deletion_signals.audit`, and a second one here that drifted by a threshold
would produce a paper disagreeing with its own appendix.

The classification, and why it is mechanical
---------------------------------------------
:func:`classify_run` is a total function of four measured criteria and the run's
stopping reason. The six labels the extension requires partition every possible
combination, evaluated in a fixed precedence, so a verdict is never a judgement
someone exercised while looking at the numbers::

    failed_run                          the run did not complete
    successful_selective_deletion       all three axes met
    deletion_without_preserved_utility  forget moved specifically; utility broke
    global_destruction                  utility broke and the damage was indiscriminate
    incomplete_forgetting               useful model, membership evidence intact
    inconclusive                        anything else, including unmeasurable

A criterion that could not be MEASURED is not satisfied. That is what keeps a
destroyed model from slipping through on a missing RNFF denominator, since
destruction and success both leave the criterion unevaluated.

Settings, seeds, and the rule against averaging
------------------------------------------------
A setting is classified only when its seeds agree. Three seeds each failing a
different criterion is not a pass, and a mean that straddles a threshold is not
a seed -- so a setting whose seeds disagree is reported as ``inconclusive`` with
every seed's own label shown beside it. Both counts the brief asks for are
reported for every setting: how many seeds passed each criterion ALONE, and how
many passed all of them TOGETHER. The second is never inferred from the first.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .reproducibility import run_provenance

FAILED_RUN = "failed_run"
SUCCESSFUL = "successful_selective_deletion"
DELETION_NO_UTILITY = "deletion_without_preserved_utility"
GLOBAL_DESTRUCTION = "global_destruction"
INCOMPLETE = "incomplete_forgetting"
INCONCLUSIVE = "inconclusive"

CLASSIFICATION_ORDER = (
    FAILED_RUN,
    SUCCESSFUL,
    DELETION_NO_UTILITY,
    GLOBAL_DESTRUCTION,
    INCOMPLETE,
    INCONCLUSIVE,
)

# Stopping reasons that mean the run did not complete as configured. A collapse
# guard trip is one: the model was destroyed mid-flight, and whatever the axes
# read afterwards describes wreckage rather than a method's capability.
FAILURE_STOP_REASONS = ("collapse_guard", "non_finite_loss", "error", "interrupted")

CONTROL_METHODS = (
    "exact_retrain",
    "null_refit",
    "routine_update",
    "exact_retrain_routine_update",
)

# Table 1's rows, in the order a reader needs them: the reference first, then
# the two things that changed nothing, then deletion, then deletion plus noise.
CONTROL_ORDER = (
    "null_refit",
    "routine_update",
    "exact_retrain",
    "exact_retrain_routine_update",
)

CENTRAL_DETECTORS = (
    "loss_ratio_vs_finetuned",
    "raw_cosine",
    "alignment_cos_sq",
    "grad_ratio",
    "sif",
    "combined_log_score",
)


# --------------------------------------------------------------- classification


@dataclass
class RunVerdict:
    """One seed's classification, with the criteria that produced it."""

    seed: int
    method: str
    setting: str
    classification: str
    membership_removed: bool
    selective_forgetting: bool
    utility_preserved: bool
    collapse_guard_not_triggered: bool
    conjunction: bool
    stop_reason: Optional[str] = None
    unevaluable: List[str] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "method": self.method,
            "setting": self.setting,
            "classification": self.classification,
            "criteria": {
                "membership_removed": self.membership_removed,
                "selective_forgetting": self.selective_forgetting,
                "utility_preserved": self.utility_preserved,
                "collapse_guard_not_triggered": self.collapse_guard_not_triggered,
            },
            "conjunction": self.conjunction,
            "stop_reason": self.stop_reason,
            "unevaluable_criteria": self.unevaluable,
            "reason": self.reason,
        }


def classify_run(run: Mapping[str, Any]) -> RunVerdict:
    """Apply the six-label partition to one audited run.

    ``run`` is an entry from ``audit_results.json["runs"]``: it carries the
    ``axes`` block the audit computed and the ``unlearning`` block the training
    loop recorded.
    """
    axes = run.get("axes") or {}
    unlearning = run.get("unlearning") or {}
    stop_reason = unlearning.get("stop_reason")

    membership = bool((axes.get("membership_removed") or {}).get("met"))
    selective = bool((axes.get("selective_forgetting") or {}).get("met"))
    utility = bool((axes.get("utility_preserved") or {}).get("met"))
    completed = stop_reason not in FAILURE_STOP_REASONS

    unevaluable = []
    if not (axes.get("utility_preserved") or {}).get("evaluable", True):
        unevaluable.append("utility_preserved")
    if not (axes.get("selective_forgetting") or {}).get("evaluable", True):
        unevaluable.append("selective_forgetting")
    if (axes.get("membership_removed") or {}).get("auc") is None:
        unevaluable.append("membership_removed")

    verdict = RunVerdict(
        seed=int(run.get("seed", -1)),
        method=str(run.get("method", "unknown")),
        setting=str(run.get("setting", "unknown")),
        classification=INCONCLUSIVE,
        membership_removed=membership,
        selective_forgetting=selective,
        utility_preserved=utility,
        collapse_guard_not_triggered=completed,
        conjunction=bool(membership and selective and utility and completed),
        stop_reason=stop_reason,
        unevaluable=unevaluable,
    )

    # Precedence, first match wins. The order is the preregistration's.
    if not completed:
        verdict.classification = FAILED_RUN
        verdict.reason = (
            f"the run stopped with {stop_reason!r} rather than completing; its "
            f"measurements are reported but describe a run that did not finish"
        )
    elif membership and selective and utility:
        verdict.classification = SUCCESSFUL
        verdict.reason = (
            "membership evidence is inside the chance-equivalence region, the "
            "forget set moved far enough relative to exact retraining, the "
            "retain set did not, and utility is intact"
        )
    elif selective and not utility:
        verdict.classification = DELETION_NO_UTILITY
        verdict.reason = (
            "the forget set moved specifically and far enough, but retained or "
            "test utility exceeded the damage threshold"
        )
    elif not utility and not selective:
        verdict.classification = GLOBAL_DESTRUCTION
        # Deliberately describes the criteria rather than the magnitude. This
        # label fires whenever utility is out of tolerance AND the collateral
        # movement is over budget, which includes a model whose perplexity rose
        # 18% as well as one that was destroyed outright. Saying "the retain set
        # moved nearly as far as the forget set" would be false of the former.
        verdict.reason = (
            "utility exceeded the damage threshold and the movement was not "
            "specific enough: collateral movement on the retain set exceeded the "
            "preregistered specificity budget. The magnitude of the damage is in "
            "the utility and RNFF columns, not in this label"
        )
    elif utility and not membership:
        verdict.classification = INCOMPLETE
        verdict.reason = (
            "the model remains useful, but forgotten records still look like "
            "training members"
        )
    else:
        verdict.reason = (
            "no preregistered label applies"
            + (f"; unmeasurable criteria: {', '.join(unevaluable)}" if unevaluable else "")
        )

    if unevaluable and verdict.classification == SUCCESSFUL:
        # Belt and braces: a criterion that was not measured cannot have been met.
        verdict.classification = INCONCLUSIVE
        verdict.reason = (
            f"criteria {', '.join(unevaluable)} could not be measured, and an "
            f"unmeasured criterion is not a satisfied one"
        )
    return verdict


def summarise_setting(verdicts: Sequence[RunVerdict]) -> Dict[str, Any]:
    """Marginal and conjunction pass counts for one setting, plus its label.

    A setting takes a classification only when every seed agrees on it. That is
    the rule the preregistration states as "do not classify a setting using only
    across-seed mean values if no individual seed passes the full conjunction",
    applied in its strongest form: disagreement is reported as disagreement.
    """
    if not verdicts:
        return {}

    labels = sorted({v.classification for v in verdicts})
    unanimous = labels[0] if len(labels) == 1 else None

    return {
        "method": verdicts[0].method,
        "setting": verdicts[0].setting,
        "n_seeds": len(verdicts),
        "seeds": sorted(v.seed for v in verdicts),
        "marginal_pass_counts": {
            "membership_removed": sum(v.membership_removed for v in verdicts),
            "selective_forgetting": sum(v.selective_forgetting for v in verdicts),
            "utility_preserved": sum(v.utility_preserved for v in verdicts),
            "collapse_guard_not_triggered": sum(
                v.collapse_guard_not_triggered for v in verdicts
            ),
        },
        "conjunction_pass_count": sum(v.conjunction for v in verdicts),
        "collapse_count": sum(not v.collapse_guard_not_triggered for v in verdicts),
        "failed_run_count": sum(v.classification == FAILED_RUN for v in verdicts),
        "per_seed_classification": {v.seed: v.classification for v in verdicts},
        "classification": unanimous or INCONCLUSIVE,
        "seeds_agree": unanimous is not None,
        "classification_note": (
            ""
            if unanimous
            else (
                f"seeds disagree ({', '.join(labels)}); reported as inconclusive "
                f"rather than averaged into a verdict"
            )
        ),
        "stop_reasons": sorted({str(v.stop_reason) for v in verdicts}),
    }


def classify_audit(audit: Mapping[str, Any]) -> Dict[str, Any]:
    """Classify every run in an audit document and group them by setting."""
    verdicts = [classify_run(run) for run in audit.get("runs", [])]

    grouped: Dict[Tuple[str, str], List[RunVerdict]] = {}
    for verdict in verdicts:
        grouped.setdefault((verdict.method, verdict.setting), []).append(verdict)

    return {
        "runs": [v.as_dict() for v in verdicts],
        "settings": [summarise_setting(group) for _, group in sorted(grouped.items())],
    }


# --------------------------------------------------------------------- tables


def strict_json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with ``None``.

    The report document is STRICT JSON, where ``null`` means "not computed" and
    there is no other way to say it. Non-finite values reach here legitimately --
    an across-seed SD is NaN when only one seed exists, and a perplexity ratio is
    NaN when its denominator was unavailable -- and ``json.dump`` with
    ``allow_nan=False`` would refuse the whole document rather than the one
    field.

    Emitting ``NaN`` instead would produce a file that is not JSON: Python reads
    it back, most other parsers do not, and a reader who got that far would have
    to guess whether it meant zero or unknown.
    """
    if isinstance(value, Mapping):
        return {key: strict_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json_safe(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "--" if not math.isfinite(as_float) else f"{as_float:.{digits}f}"


def _mean_sd(summary: Any, digits: int = 3) -> str:
    """Render an across-seed summary, or a bare number that has no spread.

    The fine-tuned reference row carries exact constants rather than summaries --
    its movement is zero by construction -- so a scalar here is a legitimate
    input, not a malformed one.
    """
    if summary is None:
        return "--"
    if not isinstance(summary, Mapping):
        return _number(summary, digits)
    if summary.get("mean") is None:
        return "--"
    mean = summary.get("mean")
    sd = summary.get("sd")
    if sd is None or (isinstance(sd, float) and not math.isfinite(sd)):
        return _number(mean, digits)
    return f"{_number(mean, digits)} ± {_number(sd, digits)}"


def _runs_by(audit: Mapping[str, Any], method: str) -> List[Mapping[str, Any]]:
    return [r for r in audit.get("runs", []) if r.get("method") == method]


def _setting_block(audit: Mapping[str, Any], method: str, setting: Optional[str] = None):
    for block in audit.get("settings", []):
        if block.get("method") == method and (setting is None or block.get("setting") == setting):
            return block
    return None


def table1_control_calibration(
    audit: Mapping[str, Any], lowprev: Optional[Mapping[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Table 1: what every detector reads when the answer is known in advance.

    The fine-tuned reference is the origin of every displacement, so its row is
    definitionally zero movement and is included to say so rather than omitted.
    """
    rows: List[Dict[str, Any]] = [
        {
            "condition": "finetuned_reference",
            "n_seeds": len(audit.get("metadata", {}).get("retrain_control_seeds", [])),
            "deletion": "none (this is the origin)",
            "membership_auc": None,
            "retain_ppl_ratio": 1.0,
            "test_ppl_ratio": 1.0,
            "rnff_forget": 0.0,
            "rnff_retain": 0.0,
            "primary_detector_auc": None,
            "note": (
                "theta_finetuned is the reference every displacement is measured "
                "from; its movement is zero by construction, not by measurement"
            ),
        }
    ]

    floors = audit.get("metadata", {}).get("null_refit_floors", {})
    for condition in CONTROL_ORDER:
        block = _setting_block(audit, condition)
        if block is None:
            rows.append(
                {"condition": condition, "n_seeds": 0, "status": "not run",
                 "note": "no score file for this condition was supplied"}
            )
            continue
        auc = block.get("auc") or {}
        metadata = audit.get("metadata", {})
        primary = metadata.get("primary_detector", "")
        # The membership axis is `loss_ratio` on forget_vs_unseen, fixed by the
        # preregistration. Read from the document when it records it, so a study
        # that changed the axis does not silently keep reporting the old one.
        membership_task = metadata.get("membership_task", "forget_vs_unseen")
        membership_score = metadata.get("membership_score", "loss_ratio")
        rows.append(
            {
                "condition": condition,
                "n_seeds": block.get("n_seeds", 0),
                "deletion": _deletion_of(condition),
                "membership_auc": (auc.get(membership_task) or {}).get(membership_score),
                "retain_ppl_ratio": (block.get("perplexity_ratio", {}) or {}).get("retain"),
                "test_ppl_ratio": (block.get("perplexity_ratio", {}) or {}).get("test"),
                "rnff_forget": (block.get("rnff", {}) or {}).get("forget"),
                "rnff_retain": (block.get("rnff", {}) or {}).get("retain"),
                "primary_detector_auc": (
                    (auc.get(audit["metadata"]["primary_task"], {}) or {}).get(primary)
                ),
                "detector_floors": _detector_floors(block, audit),
            }
        )

    return rows


def _deletion_of(condition: str) -> str:
    return {
        "null_refit": "none; only the training RNG changed",
        "routine_update": "none; an ordinary maintenance update was applied",
        "exact_retrain": "exact -- the forget set was never trained on",
        "exact_retrain_routine_update": (
            "exact, followed by the identical maintenance update"
        ),
    }.get(condition, "unknown")


def _detector_floors(block: Mapping[str, Any], audit: Mapping[str, Any]) -> Dict[str, Any]:
    """Every detector's AUC on this control, i.e. its false-positive floor."""
    task = audit.get("metadata", {}).get("primary_task", "forget_vs_retain")
    return dict(sorted((block.get("auc", {}) or {}).get(task, {}).items()))


def table2_method_verdicts(
    audit: Mapping[str, Any], classified: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    """Table 2: for each method, the selected setting and its pass counts."""
    rows: List[Dict[str, Any]] = []
    for block in classified.get("settings", []):
        method = block.get("method")
        if method in CONTROL_METHODS and method != "exact_retrain":
            continue
        counts = block.get("marginal_pass_counts", {})
        rows.append(
            {
                "method": method,
                "setting": block.get("setting"),
                "n_seeds": block.get("n_seeds"),
                "membership_pass": counts.get("membership_removed"),
                "rnff_pass": counts.get("selective_forgetting"),
                "utility_pass": counts.get("utility_preserved"),
                "conjunction_pass": block.get("conjunction_pass_count"),
                "collapse_count": block.get("collapse_count"),
                "failed_run_count": block.get("failed_run_count"),
                "classification": block.get("classification"),
                "seeds_agree": block.get("seeds_agree"),
                "per_seed_classification": block.get("per_seed_classification"),
                "note": block.get("classification_note"),
            }
        )
    return rows


def table3_detector_comparison(
    audit: Mapping[str, Any],
    benchmark: Optional[Mapping[str, Any]],
    lowprev: Optional[Mapping[str, Any]],
    setting_method: str = "exact_retrain",
    setting_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Table 3: access, cost and capability in one row per detector.

    Capability is quoted on a single named setting so the numbers in a row all
    describe the same experiment; the low-prevalence document carries every
    other setting, and the report's text says which one this is.
    """
    costs = {row["detector"]: row for row in (benchmark or {}).get("costs", [])}
    access = {row["detector"]: row for row in (benchmark or {}).get("access_table", [])}

    capability: Dict[str, Any] = {}
    for block in ((lowprev or {}).get("aggregate", {}) or {}).get("settings", []):
        if block.get("method") != setting_method:
            continue
        if setting_name is not None and block.get("setting") != setting_name:
            continue
        capability = block.get("detectors", {})
        break

    rows: List[Dict[str, Any]] = []
    for detector in CENTRAL_DETECTORS:
        spec = access.get(detector, {})
        cost = costs.get(detector, {})
        metrics = capability.get(detector, {})
        rows.append(
            {
                "detector": detector,
                "access": spec.get("access", []),
                "threat_band": spec.get("threat_band"),
                "forward_per_sample": spec.get("forward_passes_per_sample"),
                "backward_per_sample": spec.get("backward_passes_per_sample"),
                "comparable_to_baseline": spec.get(
                    "comparable_to_matched_access_baseline"
                ),
                "seconds_per_sample": cost.get("standalone_seconds_per_sample"),
                "marginal_seconds_per_sample": cost.get("marginal_seconds_per_sample"),
                "peak_gpu_mib": cost.get("peak_gpu_mib"),
                "roc_auc": (metrics.get("roc_auc") or {}),
                "pr_auc": (metrics.get("average_precision") or {}),
                "tpr_at_1pct_fpr": (metrics.get("tpr_at_fpr_0.01") or {}),
                "top_k_recall": (metrics.get("top_k_recall") or {}),
                "precision_1to100": (
                    ((metrics.get("prevalence") or {}).get("1:100") or {})
                    .get("precision_at_top_k") or {}
                ),
            }
        )
    return rows


def table4_npo_sweep(
    audit: Mapping[str, Any], classified: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    """Table 4: every NPO strength and every seed, including the failures."""
    verdicts = {
        (v["method"], v["setting"], v["seed"]): v for v in classified.get("runs", [])
    }
    rows: List[Dict[str, Any]] = []
    for run in audit.get("runs", []):
        if run.get("method") != "npo":
            continue
        axes = run.get("axes") or {}
        unlearning = run.get("unlearning") or {}
        key = (run["method"], run["setting"], run["seed"])
        rows.append(
            {
                "setting": run.get("setting"),
                "seed": run.get("seed"),
                "steps_taken": unlearning.get("steps_taken"),
                "steps_configured": unlearning.get("steps_configured"),
                "stop_reason": unlearning.get("stop_reason"),
                "membership_auc": (axes.get("membership_removed") or {}).get("auc"),
                "retain_ppl_ratio": (run.get("perplexity_ratio") or {}).get("retain"),
                "test_ppl_ratio": (run.get("perplexity_ratio") or {}).get("test"),
                "rnff_forget": (run.get("rnff") or {}).get("forget"),
                "rnff_retain": (run.get("rnff") or {}).get("retain"),
                "primary_detector_auc": (axes.get("deletion_detectable") or {}).get("auc"),
                "classification": verdicts.get(key, {}).get("classification"),
                "conjunction": verdicts.get(key, {}).get("conjunction"),
                "truncated": (run.get("data_quality") or {}).get("truncated"),
                "nonfinite": (run.get("data_quality") or {}).get("nonfinite_counts"),
            }
        )
    return sorted(rows, key=lambda r: (str(r["setting"]), r["seed"]))


# ------------------------------------------------------------------- failures


def failure_log(audit: Mapping[str, Any], classified: Mapping[str, Any],
                lowprev: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Everything that did not work, in one place, so nothing is quietly absent."""
    failed = [v for v in classified.get("runs", []) if v["classification"] == FAILED_RUN]
    incomplete = [
        {
            "path": run.get("path"),
            "method": run.get("method"),
            "setting": run.get("setting"),
            "seed": run.get("seed"),
            "missing": sorted(
                (run.get("data_quality") or {}).get("null_counts", {}).keys()
            ),
        }
        for run in audit.get("runs", [])
        if (run.get("data_quality") or {}).get("null_counts")
    ]
    return {
        "failed_runs": failed,
        "n_failed_runs": len(failed),
        "runs_with_null_score_columns": incomplete,
        "populations_not_disjoint": [
            {"path": run.get("path"), "overlaps": (run.get("data_quality") or {}).get(
                "population_overlaps")}
            for run in audit.get("runs", [])
            if not (run.get("data_quality") or {}).get("populations_disjoint", True)
        ],
        "lowprev_input_failures": (lowprev or {}).get("failures", []),
        "policy": (
            "every failed and collapsed run appears in the tables above with its "
            "classification; nothing is filtered from a summary for having failed"
        ),
    }


# -------------------------------------------------------------------- writing


def build_document(
    audit: Mapping[str, Any],
    lowprev: Optional[Mapping[str, Any]] = None,
    benchmark: Optional[Mapping[str, Any]] = None,
    table3_method: str = "exact_retrain",
    table3_setting: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble every table and the classification behind it.

    The returned document is strict-JSON safe: non-finite values are already
    ``None``. Sanitising here rather than at write time means every consumer --
    the report text, the CSVs, the figures -- sees the same document the JSON
    file contains, instead of the figures reading a NaN the file records as null.
    """
    classified = classify_audit(audit)
    document = {
        "component": "FLLM audit -- tables and verdicts",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "preregistration": audit.get("metadata", {}).get("preregistration"),
        "thresholds": audit.get("metadata", {}).get("thresholds"),
        "classification_order": list(CLASSIFICATION_ORDER),
        "classification_rule": (
            "a total function of four measured criteria and the stopping reason, "
            "evaluated in a fixed precedence; an unmeasured criterion is not a "
            "satisfied one, and a setting whose seeds disagree is reported as "
            "inconclusive rather than averaged"
        ),
        "classified": classified,
        "table1_control_calibration": table1_control_calibration(audit, lowprev),
        "table2_method_verdicts": table2_method_verdicts(audit, classified),
        "table3_detector_comparison": table3_detector_comparison(
            audit, benchmark, lowprev, table3_method, table3_setting
        ),
        "table3_setting": {"method": table3_method, "setting": table3_setting},
        "table4_npo_sweep": table4_npo_sweep(audit, classified),
        "failure_log": failure_log(audit, classified, lowprev),
        "provenance": run_provenance(),
    }
    return strict_json_safe(document)


def _write_csv(rows: Sequence[Mapping[str, Any]], path: str,
               columns: Sequence[str]) -> Optional[str]:
    if not rows:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            values = []
            for column in columns:
                value = row.get(column)
                if isinstance(value, Mapping):
                    value = value.get("mean", "")
                if isinstance(value, (list, tuple)):
                    value = "+".join(str(v) for v in value)
                values.append("" if value is None else value)
            writer.writerow(values)
    return path


def write_tables(document: Mapping[str, Any], output_dir: str) -> List[str]:
    """One CSV per table, with the columns the paper prints."""
    os.makedirs(output_dir, exist_ok=True)
    written: List[str] = []

    specs = [
        ("table1_control_calibration", "table1_control_calibration.csv",
         ("condition", "n_seeds", "deletion", "retain_ppl_ratio", "test_ppl_ratio",
          "rnff_forget", "rnff_retain", "primary_detector_auc")),
        ("table2_method_verdicts", "table2_method_verdicts.csv",
         ("method", "setting", "n_seeds", "membership_pass", "rnff_pass",
          "utility_pass", "conjunction_pass", "collapse_count", "failed_run_count",
          "classification", "seeds_agree")),
        ("table3_detector_comparison", "table3_detector_comparison.csv",
         ("detector", "threat_band", "forward_per_sample", "backward_per_sample",
          "comparable_to_baseline", "seconds_per_sample", "peak_gpu_mib",
          "roc_auc", "pr_auc", "tpr_at_1pct_fpr", "top_k_recall",
          "precision_1to100")),
        ("table4_npo_sweep", "table4_npo_sweep.csv",
         ("setting", "seed", "steps_taken", "stop_reason", "membership_auc",
          "retain_ppl_ratio", "test_ppl_ratio", "rnff_forget", "rnff_retain",
          "primary_detector_auc", "classification", "conjunction")),
    ]
    for key, filename, columns in specs:
        path = _write_csv(document.get(key, []), os.path.join(output_dir, filename), columns)
        if path:
            written.append(path)
    return written


def format_report(document: Mapping[str, Any]) -> str:
    """The Markdown report: the four tables, with their caveats attached."""
    lines: List[str] = []
    add = lines.append

    add("# Deletion-set inference under matched access — FLLM audit")
    add("")
    add(f"Generated {document['timestamp']}.")
    add(f"Preregistration: `{document.get('preregistration')}`.")
    add("")
    add("Thresholds were frozen before any run in this extension existed and are")
    add("read from that file, not from constants in code.")
    add("")

    # ---- Table 2 first: it is the verdict ------------------------------
    add("## Table 2 — Method verdicts")
    add("")
    add("`membership`, `RNFF` and `utility` count how many seeds passed each")
    add("criterion **alone**. `conjunction` counts how many passed **all of them")
    add("together on the same seed**. The second is never inferred from the first:")
    add("three seeds each failing a different criterion is not a pass.")
    add("")
    add("The classification names a **pattern of criteria**, not a magnitude.")
    add("`global_destruction` fires whenever utility is out of tolerance *and* the")
    add("collateral movement is over the specificity budget — which covers a model")
    add("whose perplexity rose 18% as well as one that was destroyed outright. Read")
    add("the magnitude off the utility and RNFF columns in Tables 1 and 4, never")
    add("off the label.")
    add("")
    add("| method | setting | seeds | membership | RNFF | utility | conjunction | collapsed | classification |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for row in document.get("table2_method_verdicts", []):
        add(
            f"| {row['method']} | {row['setting']} | {row['n_seeds']} | "
            f"{row['membership_pass']}/{row['n_seeds']} | "
            f"{row['rnff_pass']}/{row['n_seeds']} | "
            f"{row['utility_pass']}/{row['n_seeds']} | "
            f"**{row['conjunction_pass']}/{row['n_seeds']}** | "
            f"{row['collapse_count']} | {row['classification']} |"
        )
    add("")
    disagreeing = [r for r in document.get("table2_method_verdicts", []) if not r.get("seeds_agree")]
    if disagreeing:
        add("Settings whose seeds disagreed, shown per seed rather than averaged:")
        add("")
        for row in disagreeing:
            per_seed = ", ".join(
                f"seed {s}: {c}" for s, c in sorted(row["per_seed_classification"].items())
            )
            add(f"- `{row['method']} / {row['setting']}` — {per_seed}")
        add("")

    # ---- Table 1 -------------------------------------------------------
    add("## Table 1 — Control calibration")
    add("")
    add("What the axes and the detectors read when the answer is known in advance.")
    add("`null_refit` and `routine_update` deleted **nothing**, so anything they")
    add("produce is a false positive; the first measures training randomness and")
    add("the second measures ordinary maintenance.")
    add("")
    add("| condition | seeds | deletion | retain ppl | test ppl | RNFF forget | RNFF retain |")
    add("|---|---:|---|---|---|---|---|")
    for row in document.get("table1_control_calibration", []):
        if row.get("status") == "not run":
            add(f"| {row['condition']} | 0 | — | — | — | — | — |")
            continue
        add(
            f"| {row['condition']} | {row.get('n_seeds', 0)} | "
            f"{row.get('deletion', '')} | "
            f"{_mean_sd(row.get('retain_ppl_ratio'))} | "
            f"{_mean_sd(row.get('test_ppl_ratio'))} | "
            f"{_mean_sd(row.get('rnff_forget'))} | "
            f"{_mean_sd(row.get('rnff_retain'))} |"
        )
    add("")

    # ---- Table 3 -------------------------------------------------------
    table3_setting = document.get("table3_setting", {})
    add("## Table 3 — Matched-access detector comparison")
    add("")
    add(
        f"Capability is quoted on `{table3_setting.get('method')}"
        f"{'/' + table3_setting['setting'] if table3_setting.get('setting') else ''}`, "
        "so every number in a row describes the same experiment."
    )
    add("")
    add("**`comparable`** is the matched-access rule: a detector may be claimed")
    add("advantageous over the two-checkpoint loss baseline only if it already")
    add("holds everything that baseline holds. A `no` means the row is reported")
    add("for completeness and supports no comparative claim.")
    add("")
    add("| detector | threat band | fwd | bwd | comparable | s/sample | peak MiB | ROC-AUC | PR-AUC | TPR@1%FPR | top-k recall |")
    add("|---|---|---:|---:|:--:|---:|---:|---|---|---|---|")
    for row in document.get("table3_detector_comparison", []):
        add(
            f"| `{row['detector']}` | {row.get('threat_band', '')} | "
            f"{row.get('forward_per_sample', '')} | {row.get('backward_per_sample', '')} | "
            f"{'yes' if row.get('comparable_to_baseline') else 'no'} | "
            f"{_number(row.get('seconds_per_sample'), 4)} | "
            f"{_number(row.get('peak_gpu_mib'), 0)} | "
            f"{_mean_sd(row.get('roc_auc'))} | {_mean_sd(row.get('pr_auc'))} | "
            f"{_mean_sd(row.get('tpr_at_1pct_fpr'))} | "
            f"{_mean_sd(row.get('top_k_recall'))} |"
        )
    add("")

    # ---- Table 4 -------------------------------------------------------
    npo_rows = document.get("table4_npo_sweep", [])
    add("## Table 4 — NPO strength sweep")
    add("")
    if not npo_rows:
        add("_No NPO runs were supplied._")
    else:
        add("Every strength and every seed, including collapsed and failed runs.")
        add("")
        add("| setting | seed | steps | stop | membership | retain ppl | RNFF forget | RNFF retain | classification |")
        add("|---|---:|---:|---|---|---|---|---|---|")
        for row in npo_rows:
            add(
                f"| {row['setting']} | {row['seed']} | {row.get('steps_taken', '')} | "
                f"{row.get('stop_reason', '')} | {_number(row.get('membership_auc'))} | "
                f"{_number(row.get('retain_ppl_ratio'))} | "
                f"{_number(row.get('rnff_forget'))} | {_number(row.get('rnff_retain'))} | "
                f"{row.get('classification', '')} |"
            )
    add("")

    # ---- failures ------------------------------------------------------
    failures = document.get("failure_log", {})
    add("## Failure log")
    add("")
    add(f"Failed or collapsed runs: **{failures.get('n_failed_runs', 0)}**. "
        "All of them appear in the tables above.")
    if failures.get("populations_not_disjoint"):
        add("")
        add("**Populations not disjoint in:**")
        for entry in failures["populations_not_disjoint"]:
            add(f"- `{entry['path']}` — {entry['overlaps']}")
    if failures.get("lowprev_input_failures"):
        add("")
        add("**Low-prevalence inputs that could not be read:**")
        for entry in failures["lowprev_input_failures"]:
            add(f"- `{entry['path']}` — {entry['error']}")
    add("")

    add("## Caveats that apply to every number above")
    add("")
    add("- **ArxivMIA, Llama-3.2-1B, LoRA, and these methods only.** Nothing here")
    add("  licenses a claim beyond that scope.")
    add("- **n = 3 seeds.** Every seed is shown. Means, paired within-seed")
    add("  differences and across-seed SDs are the evidence; n=3 does not provide")
    add("  population-level statistical certainty.")
    add("- **Membership AUC near chance is not proof of forgetting.** It is one of")
    add("  four criteria and never certifies anything alone; a destroyed model")
    add("  reaches chance by destroying the signal along with the model.")
    add("- **Access.** Every two-checkpoint detector requires the pre-deletion")
    add("  checkpoint. These are audit capabilities for a model owner, regulator")
    add("  or authorised verifier — not black-box membership-inference attacks,")
    add("  and detecting a deletion is not by itself a privacy harm.")
    add("- **Bootstrap intervals** are over samples within one run. They describe")
    add("  that run's precision, never variation between independently trained")
    add("  models.")
    add("")
    return "\n".join(lines)


# -------------------------------------------------------------------- the CLI


def _load(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    if not os.path.exists(path):
        raise SystemExit(f"No such file: {path}")
    with open(path, "r") as handle:
        return json.load(handle)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble the FLLM audit's four tables and its report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  $PY -m deletion_signals.fllm_report \\\n"
            "      --audit fllm_audit/reports/audit_results.json \\\n"
            "      --lowprev fllm_audit/lowprev/lowprev_results.json \\\n"
            "      --benchmark fllm_audit/benchmark/benchmark.json \\\n"
            "      --output fllm_audit/reports\n"
        ),
    )
    parser.add_argument("--audit", required=True)
    parser.add_argument("--lowprev", default=None)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--table3-method", default="exact_retrain",
                        help="Which setting Table 3 quotes capability on")
    parser.add_argument("--table3-setting", default=None)
    parser.add_argument("--output", default="./fllm_audit/reports")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    document = build_document(
        audit=_load(args.audit),
        lowprev=_load(args.lowprev),
        benchmark=_load(args.benchmark),
        table3_method=args.table3_method,
        table3_setting=args.table3_setting,
    )

    os.makedirs(args.output, exist_ok=True)
    json_path = os.path.join(args.output, "fllm_tables.json")
    with open(json_path, "w") as handle:
        json.dump(document, handle, indent=2, allow_nan=False, default=str)

    report_path = os.path.join(args.output, "fllm_report.md")
    with open(report_path, "w") as handle:
        handle.write(format_report(document))

    failure_path = os.path.join(args.output, "failure_log.json")
    with open(failure_path, "w") as handle:
        json.dump(document["failure_log"], handle, indent=2, allow_nan=False, default=str)

    written = write_tables(document, args.output)

    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {failure_path}")
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
