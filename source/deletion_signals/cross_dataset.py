"""The cross-dataset story: where every metric's null actually sits.

Why this module exists
----------------------
The ArxivMIA and BookMIA studies each produced a complete report, and neither
tells the story, because the story is the **comparison between them**. Four
metrics are used to judge unlearning; for each one, the null a practitioner
assumes is not the null the data show, and it takes a specific control to find
out where the null really is:

===================  =====================  ===============================
metric               assumed null           control that locates the real one
===================  =====================  ===============================
membership AUC       0.5                    exact retraining
RNFF                 0 with nothing deleted routine update
detector AUC         comparable detectors   the matched-access rule
detector AUC         reflects capability    prevalence adjustment
===================  =====================  ===============================

This module assembles that table from the two studies' outputs, then reports
the object-level consequence: with all four controls installed, no approximate
method passes on either corpus.

The sensitivity analysis, and what it is not
--------------------------------------------
:func:`sensitivity_to_membership_null` re-decides every verdict using each
corpus's **measured** null -- the membership AUC that exact retraining, a
perfect deletion, actually produces -- instead of the preregistered absolute
region. It exists to answer "does this matter?" with a count of changed
verdicts rather than an argument.

It is **post-hoc and secondary**. It does not replace the preregistered verdict,
it is never reported alone, and the preregistered region is not recentred in the
primary tables. Reporting both is the point: the gap between them is the finding.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .reproducibility import run_provenance
from .fllm_report import (
    CLASSIFICATION_ORDER,
    FAILED_RUN,
    INCOMPLETE,
    RunVerdict,
    classify_run,
    strict_json_safe,
    summarise_setting,
)

# The half-width of the preregistered membership equivalence region, [0.45,
# 0.55]. Reused when the region is recentred on a measured null, so the
# sensitivity analysis changes only the CENTRE and not the tolerance.
REGION_HALF_WIDTH = 0.05


@dataclass
class Study:
    """One dataset/model study's outputs."""

    name: str
    dataset: str
    model: str
    audit: Dict[str, Any]
    lowprev: Optional[Dict[str, Any]] = None
    tables: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)

    @property
    def runs(self) -> List[Dict[str, Any]]:
        return self.audit.get("runs", [])


# --------------------------------------------------------------- the four nulls


def measured_membership_null(study: Study) -> Dict[str, Any]:
    """Where the membership axis actually reads on a PERFECT deletion.

    Exact retraining never trained on the forget set, so a calibrated axis must
    read chance on it. Whatever it reads instead is that corpus's empirical
    null, and every membership number on that corpus has to be read against it.
    """
    values = [
        (run.get("axes", {}).get("membership_removed") or {}).get("auc")
        for run in study.runs
        if run.get("method") == "exact_retrain"
    ]
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return {"measured": None, "n_seeds": 0,
                "note": "no exact-retraining control in this study"}
    mean = statistics.mean(values)
    return {
        "measured": mean,
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "n_seeds": len(values),
        "assumed": 0.5,
        "deviation": abs(mean - 0.5),
        "inside_preregistered_region": 0.45 <= mean <= 0.55,
    }


def measured_rnff_null(study: Study) -> Dict[str, Any]:
    """What RNFF reads when NOTHING was deleted.

    Two conditions, and they differ. The null refit changes only training
    randomness; the routine update applies an ordinary, legitimate maintenance
    pass. A practitioner assumes both read zero.
    """
    out: Dict[str, Any] = {"assumed": 0.0}
    for condition in ("null_refit", "routine_update"):
        values = [
            (run.get("rnff") or {}).get("forget")
            for run in study.runs
            if run.get("method") == condition
        ]
        values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
        if values:
            out[condition] = {
                "measured": statistics.mean(values),
                "sd": statistics.stdev(values) if len(values) > 1 else None,
                "n_seeds": len(values),
            }
    return out


def measured_prevalence_gap(study: Study, method: str = "exact_retrain") -> Dict[str, Any]:
    """How far AUC overstates operational capability at 1:100 prevalence."""
    if not study.lowprev:
        return {}
    settings = (study.lowprev.get("aggregate", {}) or {}).get("settings", [])
    block = next((s for s in settings if s.get("method") == method), None)
    if not block:
        return {}

    rows = {}
    for detector, entry in block.get("detectors", {}).items():
        auc = (entry.get("roc_auc") or {}).get("mean")
        precision = (
            ((entry.get("prevalence") or {}).get("1:100") or {})
            .get("precision_at_top_k") or {}
        ).get("mean")
        if auc is None or precision is None:
            continue
        rows[detector] = {
            "roc_auc": auc,
            "precision_at_1to100": precision,
            "gap": auc - precision,
        }
    return {"setting": method, "detectors": rows}


def matched_access_result(study: Study, baseline: str = "loss_ratio_vs_finetuned",
                          method: str = "exact_retrain") -> Dict[str, Any]:
    """Paired across-seed differences against the matched-access baseline."""
    from .benchmark import comparable

    block = next(
        (s for s in study.audit.get("settings", []) if s.get("method") == method), None
    )
    if not block:
        return {}

    rows = {}
    for entry in block.get("paired_differences", []):
        other = entry.get("other")
        if other is None:
            continue
        try:
            is_comparable = comparable(other, baseline)
        except KeyError:
            continue
        mean, sd = entry.get("mean"), entry.get("sd") or 0.0
        rows[other] = {
            "mean_diff": mean,
            "sd": sd,
            "comparable": is_comparable,
            "verdict": (
                "not comparable"
                if not is_comparable
                else "baseline wins" if mean > abs(sd)
                else "challenger wins" if mean < -abs(sd)
                else "no separation"
            ),
        }
    return {"baseline": baseline, "setting": method, "challengers": rows}


def where_is_chance(studies: Sequence[Study]) -> Dict[str, Any]:
    """The paper's central table: four metrics, four nulls, four controls."""
    return {
        "membership_auc": {
            "assumed_null": 0.5,
            "control_that_locates_it": "exact retraining",
            "per_study": {s.name: measured_membership_null(s) for s in studies},
        },
        "rnff": {
            "assumed_null": 0.0,
            "control_that_locates_it": "null refit and routine update",
            "per_study": {s.name: measured_rnff_null(s) for s in studies},
        },
        "detector_auc_comparability": {
            "assumed": "detectors are comparable to one another",
            "control_that_locates_it": "the matched-access rule",
            "per_study": {s.name: matched_access_result(s) for s in studies},
        },
        "detector_auc_capability": {
            "assumed": "AUC reflects operational capability",
            "control_that_locates_it": "prevalence adjustment",
            "per_study": {s.name: measured_prevalence_gap(s) for s in studies},
        },
    }


# ------------------------------------------------------- the sensitivity analysis


def sensitivity_to_membership_null(study: Study) -> Dict[str, Any]:
    """Re-decide every verdict against this corpus's MEASURED membership null.

    Only the centre of the equivalence region moves, from 0.5 to whatever exact
    retraining reads; the half-width stays at the preregistered 0.05. Everything
    else -- RNFF, utility, the collapse guard, the classification precedence --
    is untouched.

    POST-HOC AND SECONDARY. Reported beside the preregistered verdict, never
    instead of it.
    """
    null = measured_membership_null(study)
    if null.get("measured") is None:
        return {"applicable": False, "reason": "no exact-retraining control"}

    centre = float(null["measured"])
    low, high = centre - REGION_HALF_WIDTH, centre + REGION_HALF_WIDTH

    changed: List[Dict[str, Any]] = []
    recomputed: Dict[Tuple[str, str], List[RunVerdict]] = {}

    for run in study.runs:
        preregistered = classify_run(run)

        auc = (run.get("axes", {}).get("membership_removed") or {}).get("auc")
        membership = (
            auc is not None and math.isfinite(float(auc)) and low <= float(auc) <= high
        )

        # Re-run the same classifier with only the membership bit substituted.
        adjusted = dict(run)
        axes = dict(run.get("axes") or {})
        axes["membership_removed"] = {
            **(axes.get("membership_removed") or {}),
            "met": membership,
        }
        adjusted["axes"] = axes
        verdict = classify_run(adjusted)

        recomputed.setdefault((verdict.method, verdict.setting), []).append(verdict)

        if verdict.classification != preregistered.classification:
            changed.append(
                {
                    "method": verdict.method,
                    "setting": verdict.setting,
                    "seed": verdict.seed,
                    "membership_auc": auc,
                    "preregistered": preregistered.classification,
                    "under_measured_null": verdict.classification,
                }
            )

    settings = [summarise_setting(group) for _, group in sorted(recomputed.items())]

    return {
        "applicable": True,
        "status": "POST-HOC SENSITIVITY -- secondary to the preregistered verdict",
        "measured_null": centre,
        "region_used": [low, high],
        "region_half_width": REGION_HALF_WIDTH,
        "preregistered_region": [0.45, 0.55],
        "what_changed": (
            "only the CENTRE of the membership equivalence region; the "
            "half-width, the other three criteria and the classification "
            "precedence are untouched"
        ),
        "n_runs_reclassified": len(changed),
        "reclassified": changed,
        "settings": settings,
        "conjunction_passes_under_measured_null": sum(
            s.get("conjunction_pass_count", 0) for s in settings
        ),
    }


# ------------------------------------------------------------------ the report


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "--" if not math.isfinite(as_float) else f"{as_float:.{digits}f}"


def _mean_sd(entry: Optional[Mapping[str, Any]], key: str = "measured",
             digits: int = 3) -> str:
    if not entry or entry.get(key) is None:
        return "--"
    sd = entry.get("sd")
    if sd is None:
        return _fmt(entry[key], digits)
    return f"{_fmt(entry[key], digits)} ± {_fmt(sd, digits)}"


def format_report(document: Mapping[str, Any]) -> str:
    lines: List[str] = []
    add = lines.append
    studies = document["studies"]
    chance = document["where_is_chance"]

    add("# Where is chance? Calibration controls for unlearning evaluation")
    add("")
    add(f"Generated {document['timestamp']}.")
    add("")
    add("| study | dataset | model | seeds |")
    add("|---|---|---|---:|")
    for s in studies:
        add(f"| {s['name']} | `{s['dataset']}` | `{s['model']}` | {s['n_seeds']} |")
    add("")

    # ---- the central table --------------------------------------------
    add("## Table A — Four metrics, four nulls, four controls")
    add("")
    add("Each row is a quantity used to judge unlearning, the null a practitioner")
    add("assumes it has, and where the null actually sits once the corresponding")
    add("control is run.")
    add("")
    add("### A1. Membership AUC — the null is not 0.5, and it moves between corpora")
    add("")
    add("Measured on **exact retraining**: the forget set was never trained on, so")
    add("a calibrated axis must read chance. Whatever it reads instead is that")
    add("corpus's empirical null.")
    add("")
    add("| study | measured null | assumed | deviation | inside [0.45, 0.55]? |")
    add("|---|---|---:|---:|:--:|")
    for s in studies:
        entry = chance["membership_auc"]["per_study"].get(s["name"], {})
        add(
            f"| {s['name']} | {_mean_sd(entry)} | 0.500 | "
            f"{_fmt(entry.get('deviation'))} | "
            f"{'yes' if entry.get('inside_preregistered_region') else '**no**'} |"
        )
    add("")

    add("### A2. RNFF — nonzero when nothing was deleted")
    add("")
    add("| study | null refit | routine update |")
    add("|---|---|---|")
    for s in studies:
        entry = chance["rnff"]["per_study"].get(s["name"], {})
        add(f"| {s['name']} | {_mean_sd(entry.get('null_refit'))} | "
            f"{_mean_sd(entry.get('routine_update'))} |")
    add("")
    add("Both conditions deleted **nothing**. A detector or metric reading away")
    add("from zero here is producing a false signal.")
    add("")

    add("### A3. Detector AUC — comparability requires matched access")
    add("")
    add("Paired within-seed difference, AUC(two-checkpoint loss) − AUC(challenger),")
    add("on exact retraining. Positive means the loss baseline wins.")
    add("")
    header = "| challenger | " + " | ".join(s["name"] for s in studies) + " |"
    add(header)
    add("|---" * (len(studies) + 1) + "|")
    all_challengers = sorted({
        c
        for s in studies
        for c in (chance["detector_auc_comparability"]["per_study"]
                  .get(s["name"], {}).get("challengers", {}))
    })
    for challenger in all_challengers:
        cells = []
        for s in studies:
            row = (chance["detector_auc_comparability"]["per_study"]
                   .get(s["name"], {}).get("challengers", {}).get(challenger))
            if not row:
                cells.append("--")
            elif not row["comparable"]:
                cells.append("_not comparable_")
            else:
                cells.append(f"{row['mean_diff']:+.4f} ± {row['sd']:.4f}")
        add(f"| `{challenger}` | " + " | ".join(cells) + " |")
    add("")

    add("### A4. Detector AUC — capability collapses at realistic prevalence")
    add("")
    add("| study | detector | ROC-AUC | precision at 1:100 |")
    add("|---|---|---|---|")
    for s in studies:
        gap = chance["detector_auc_capability"]["per_study"].get(s["name"], {})
        for detector, row in sorted(
            (gap.get("detectors") or {}).items(), key=lambda kv: -kv[1]["roc_auc"]
        ):
            add(f"| {s['name']} | `{detector}` | {_fmt(row['roc_auc'], 4)} | "
                f"{_fmt(row['precision_at_1to100'], 4)} |")
    add("")

    # ---- the object-level result --------------------------------------
    add("## Table B — With the controls installed, what passes")
    add("")
    add("Conjunction: membership removed **and** selective forgetting **and**")
    add("utility preserved **and** the collapse guard not triggered, on the same")
    add("independently trained seed.")
    add("")
    add("| study | method | setting | conjunction | classification |")
    add("|---|---|---|---:|---|")
    for s in studies:
        for row in s["verdicts"]:
            add(f"| {s['name']} | {row['method']} | {row['setting']} | "
                f"**{row['conjunction_pass_count']}/{row['n_seeds']}** | "
                f"{row['classification']} |")
    add("")

    # ---- sensitivity ---------------------------------------------------
    add("## Table C — Sensitivity: does the membership null matter?")
    add("")
    add("**Post-hoc and secondary.** Re-decides every verdict with the membership")
    add("equivalence region recentred on that corpus's *measured* null instead of")
    add("0.5. Only the centre moves; the half-width, the other three criteria and")
    add("the classification precedence are untouched. The preregistered verdict in")
    add("Table B stands.")
    add("")
    add("| study | measured null | region used | runs reclassified | conjunction passes |")
    add("|---|---|---|---:|---:|")
    for s in studies:
        sens = s.get("sensitivity", {})
        if not sens.get("applicable"):
            add(f"| {s['name']} | -- | -- | -- | -- |")
            continue
        region = sens["region_used"]
        add(f"| {s['name']} | {_fmt(sens['measured_null'])} | "
            f"[{region[0]:.3f}, {region[1]:.3f}] | {sens['n_runs_reclassified']} | "
            f"{sens['conjunction_passes_under_measured_null']} |")
    add("")
    for s in studies:
        sens = s.get("sensitivity", {})
        for entry in sens.get("reclassified", []):
            add(f"- **{s['name']}** `{entry['method']} / {entry['setting']}` seed "
                f"{entry['seed']} (membership {_fmt(entry['membership_auc'])}): "
                f"`{entry['preregistered']}` → `{entry['under_measured_null']}`")
    add("")

    add("## What this licenses, and what it does not")
    add("")
    add("- **Licensed:** that these four metrics cannot be read without the")
    add("  controls that locate their nulls, on the corpora and models tested;")
    add("  and that no method tested here passes the conjunction on either.")
    add("- **Not licensed:** any claim about approximate unlearning in general,")
    add("  about untested corpora or architectures, or about methods and settings")
    add("  not run. n = 3 seeds throughout.")
    add("- **Confounded:** the two studies differ in corpus, model family **and**")
    add("  truncation regime simultaneously (100% of BookMIA samples are")
    add("  truncated at 512 tokens against a corpus mean of 710; ArxivMIA")
    add("  truncates almost nothing at a mean of 205). No cross-dataset")
    add("  difference is attributed to any one of the three.")
    add("")
    return "\n".join(lines)


# ------------------------------------------------------------------- assembly


def build_document(studies: Sequence[Study]) -> Dict[str, Any]:
    from .fllm_report import classify_audit

    payload_studies = []
    for study in studies:
        classified = classify_audit(study.audit)
        payload_studies.append(
            {
                "name": study.name,
                "dataset": study.dataset,
                "model": study.model,
                "n_seeds": len({r.get("seed") for r in study.runs}),
                "verdicts": [
                    b for b in classified["settings"]
                    if b.get("method") not in ("null_refit", "routine_update",
                                               "exact_retrain_routine_update")
                ],
                "sensitivity": sensitivity_to_membership_null(study),
                "notes": study.notes,
            }
        )

    return strict_json_safe(
        {
            "title": "Where is chance? Calibration controls for unlearning evaluation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "thesis": (
                "Every metric used to judge approximate unlearning has a null "
                "that practitioners assume and that the data contradict, and "
                "each one requires a specific control to locate. With all four "
                "controls installed, no method tested passes."
            ),
            "studies": payload_studies,
            "where_is_chance": where_is_chance(studies),
            "provenance": run_provenance(),
        }
    )


def load_study(name: str, root: str, dataset: str, model: str,
               notes: Optional[List[str]] = None) -> Study:
    def read(path: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(path):
            return None
        with open(path) as handle:
            return json.load(handle)

    audit = read(os.path.join(root, "reports", "audit_results.json"))
    if audit is None:
        raise SystemExit(f"{name}: no audit_results.json under {root}/reports")
    return Study(
        name=name,
        dataset=dataset,
        model=model,
        audit=audit,
        lowprev=read(os.path.join(root, "lowprev", "lowprev_results.json")),
        tables=read(os.path.join(root, "reports", "fllm_tables.json")),
        notes=notes or [],
    )


def write_csv(document: Mapping[str, Any], path: str) -> str:
    """The Table A1 row that carries the paper: the null, per corpus."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    chance = document["where_is_chance"]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["study", "metric", "assumed_null", "measured_null",
                         "sd", "n_seeds", "deviation"])
        for name, entry in chance["membership_auc"]["per_study"].items():
            writer.writerow([name, "membership_auc", 0.5,
                             entry.get("measured"), entry.get("sd"),
                             entry.get("n_seeds"), entry.get("deviation")])
        for name, entry in chance["rnff"]["per_study"].items():
            for condition in ("null_refit", "routine_update"):
                block = entry.get(condition)
                if block:
                    writer.writerow([name, f"rnff_{condition}", 0.0,
                                     block.get("measured"), block.get("sd"),
                                     block.get("n_seeds"),
                                     abs(block.get("measured", 0.0))])
    return path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate the ArxivMIA and BookMIA studies into one report.",
    )
    parser.add_argument("--study", action="append", nargs=4,
                        metavar=("NAME", "ROOT", "DATASET", "MODEL"),
                        help="Repeatable: display name, output root, dataset, model")
    parser.add_argument("--output", default="./ds_cross_dataset")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    specs = args.study or [
        ["ArxivMIA / Llama-3.2-1B", "fllm_audit", "zhliu/ArxivMIA",
         "meta-llama/Llama-3.2-1B"],
        ["BookMIA / Qwen2.5-1.5B", "ds_bookmia", "swj0419/BookMIA",
         "Qwen/Qwen2.5-1.5B"],
    ]
    studies = [load_study(name, root, dataset, model)
               for name, root, dataset, model in specs]

    document = build_document(studies)
    os.makedirs(args.output, exist_ok=True)

    json_path = os.path.join(args.output, "cross_dataset.json")
    with open(json_path, "w") as handle:
        json.dump(document, handle, indent=2, allow_nan=False, default=str)
    report_path = os.path.join(args.output, "cross_dataset_report.md")
    with open(report_path, "w") as handle:
        handle.write(format_report(document))
    csv_path = write_csv(document, os.path.join(args.output, "where_is_chance.csv"))

    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
