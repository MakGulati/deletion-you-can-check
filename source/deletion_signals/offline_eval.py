"""Offline evaluation: metrics, intervals, paired comparisons, correlations.

Reads saved per-sample score tables and recomputes everything. No model is
loaded and no GPU is touched, so changing an FPR target, adding a task or
correcting a test is seconds of CPU rather than another collection run.

Every AUC uses the orientation declared in ``configs/orientations.yaml``. There
is no code path in this module that computes ``max(auc, 1 - auc)``: a score that
lands below 0.5 under its declared direction is reported below 0.5.

Two units of analysis, kept apart:

* per-run numbers carry a bootstrap interval over samples -- the precision of
  the estimate given this one trained model;
* the across-seed block treats each seed as one observation and is the only
  place a claim like "alignment beats the gradient norm for GA+GD" can be
  supported from.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .config import load_config
from .orientations import OrientationTable, build_task_arrays
from .reproducibility import new_rng, run_provenance
from .schema import SCORE_COLUMNS, ScoreFile, load_any
from .stats import (
    across_seed_comparison,
    average_precision,
    bootstrap_auc_ci,
    paired_bootstrap_auc_diff,
    spearman,
    summarise_across_seeds,
    tpr_at_fpr,
)

# The comparison the study exists to make. Every other score is measured against
# the FIRST entry on the primary task, so the head of this list is the reference
# the paired within-run and across-seed tables compare everything else to.
#
# `raw_cosine` heads it as of the GA+GD unlearning-strength sweep: the signed
# cosine, not its square, is the preregistered primary score there, for the
# reasons written into configs/orientations.yaml. `alignment_cos_sq` stays as
# the first baseline -- the difference between the two is exactly the sign
# information, so the comparison measures what squaring costs.
#
# `combined_log_score` is a baseline and is never presented as primary.
PRIMARY_COMPARISON = [
    "raw_cosine",
    "alignment_cos_sq",
    "grad_norm_sq",
    "grad_ratio",
    "log_grad_ratio",
    "fuma_lora_ratio",
    "loss_ratio",
    "loss_delta",
    "loss_ratio_vs_finetuned",
    "loss_delta_vs_finetuned",
    "loss_unlearned",
    "sif",
    "combined_log_score",
]


def _run_key(method: str, seed: int) -> str:
    return f"{method}/seed{seed}"


def group_rows(
    rows: Sequence[Mapping[str, Any]]
) -> Dict[Tuple[str, int], List[Mapping[str, Any]]]:
    """Partition rows by (method, seed) -- one trained-and-unlearned model each."""
    grouped: Dict[Tuple[str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("method", "unknown")), int(row.get("seed", -1)))].append(row)
    return dict(grouped)


def evaluate_run(
    rows: Sequence[Mapping[str, Any]],
    table: OrientationTable,
    tasks: Sequence[str],
    scores: Sequence[str],
    fpr_targets: Sequence[float],
    n_bootstrap: int,
    confidence: float,
    rng_seed: int,
) -> Dict[str, Any]:
    """All metrics for one (method, seed) run."""
    results: Dict[str, Any] = {}

    for task_name in tasks:
        task = table.task(task_name)
        task_block: Dict[str, Any] = {
            "positive": list(task.positive),
            "negative": list(task.negative),
            "role": task.role,
            "scores": {},
        }

        for score in scores:
            values, labels, dropped = build_task_arrays(rows, score, task)
            if not values or len(set(labels)) < 2:
                task_block["scores"][score] = {
                    "available": False,
                    "reason": (
                        "no finite values for this score in this run"
                        if not values
                        else "only one class present"
                    ),
                    "n_dropped": dropped,
                }
                continue

            orientation = table.orientation(score, task_name)
            oriented = table.orient(values, score, task_name)

            estimate = bootstrap_auc_ci(
                oriented,
                labels,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                rng=new_rng(rng_seed),
            )
            entry: Dict[str, Any] = {
                "available": True,
                "direction": orientation.direction,
                "headline": orientation.headline,
                **estimate.as_dict(),
                "average_precision": average_precision(oriented, labels),
                "n_dropped": dropped,
                "tpr_at_fpr": {
                    f"{target:g}": tpr_at_fpr(oriented, labels, target).as_dict()
                    for target in fpr_targets
                },
            }
            task_block["scores"][score] = entry

        results[task_name] = task_block

    return results


def paired_within_run(
    rows: Sequence[Mapping[str, Any]],
    table: OrientationTable,
    task_name: str,
    reference: str,
    others: Sequence[str],
    n_bootstrap: int,
    rng_seed: int,
) -> Dict[str, Any]:
    """Within-run paired bootstrap of reference-vs-other on identical samples.

    Restricted to samples where BOTH scores are available, so the comparison is
    genuinely paired: a FUMA ratio missing on half the rows would otherwise be
    compared against alignment on a different sample set.
    """
    task = table.task(task_name)
    comparisons: Dict[str, Any] = {}

    for other in others:
        if other == reference:
            continue
        paired_values_a: List[float] = []
        paired_values_b: List[float] = []
        paired_labels: List[int] = []

        for row in rows:
            label = task.label_of(str(row.get("population")))
            if label is None:
                continue
            a, b = row.get(reference), row.get(other)
            if a is None or b is None:
                continue
            try:
                a_f, b_f = float(a), float(b)
            except (TypeError, ValueError):
                continue
            if math.isnan(a_f) or math.isnan(b_f):
                continue
            paired_values_a.append(a_f)
            paired_values_b.append(b_f)
            paired_labels.append(label)

        if len(set(paired_labels)) < 2:
            comparisons[other] = {"available": False, "reason": "no paired samples"}
            continue

        comparisons[other] = paired_bootstrap_auc_diff(
            table.orient(paired_values_a, reference, task_name),
            table.orient(paired_values_b, other, task_name),
            paired_labels,
            n_bootstrap=n_bootstrap,
            rng=new_rng(rng_seed),
        )
        comparisons[other]["n_paired"] = len(paired_labels)

    return comparisons


def correlation_block(
    rows: Sequence[Mapping[str, Any]], pairs: Sequence[Sequence[str]]
) -> Dict[str, Any]:
    """Spearman correlations between score pairs, over all evaluated samples.

    Diagnostic, not inferential: a near-1 correlation between the self-influence
    score and the squared gradient norm means the curvature term is inert, which
    is a fact about the estimator rather than about the data.
    """
    block: Dict[str, Any] = {}
    for pair in pairs:
        a, b = pair[0], pair[1]
        xs = [row.get(a) for row in rows]
        ys = [row.get(b) for row in rows]
        block[f"{a}~{b}"] = spearman(
            [x for x, y in zip(xs, ys) if x is not None and y is not None],
            [y for x, y in zip(xs, ys) if x is not None and y is not None],
        )
    return block


def evaluate(
    files: Sequence[ScoreFile],
    config: Mapping[str, Any],
    table: OrientationTable,
) -> Dict[str, Any]:
    """The full offline evaluation over a set of score files."""
    evaluation = config["evaluation"]
    tasks = list(evaluation["tasks"])
    primary_task = evaluation["primary_task"]
    fpr_targets = list(evaluation["fpr_targets"])
    n_bootstrap = int(evaluation["n_bootstrap"])
    confidence = float(evaluation["confidence"])

    all_rows = [row for score_file in files for row in score_file.rows]
    if not all_rows:
        raise ValueError("No rows were loaded; nothing to evaluate.")

    grouped = group_rows(all_rows)
    available = sorted({s for f in files for s in f.available_scores})
    scores = [s for s in SCORE_COLUMNS if s in available]
    absent = [s for s in SCORE_COLUMNS if s not in available]

    per_run: Dict[str, Any] = {}
    paired_runs: Dict[str, Any] = {}
    correlations: Dict[str, Any] = {}
    auc_by_method_task_score: Dict[Tuple[str, str, str], Dict[int, float]] = defaultdict(dict)

    for (method, seed), rows in sorted(grouped.items()):
        key = _run_key(method, seed)
        run_results = evaluate_run(
            rows,
            table,
            tasks,
            scores,
            fpr_targets,
            n_bootstrap,
            confidence,
            rng_seed=seed,
        )
        per_run[key] = {"method": method, "seed": seed, "n_rows": len(rows), "tasks": run_results}

        for task_name, task_block in run_results.items():
            for score, entry in task_block["scores"].items():
                if entry.get("available"):
                    auc_by_method_task_score[(method, task_name, score)][seed] = entry["auc"]

        reference = PRIMARY_COMPARISON[0]
        if reference in scores:
            paired_runs[key] = paired_within_run(
                rows,
                table,
                primary_task,
                reference,
                [s for s in PRIMARY_COMPARISON if s in scores],
                n_bootstrap=n_bootstrap,
                rng_seed=seed,
            )

        correlations[key] = correlation_block(rows, evaluation["correlations"])

    across_seeds: Dict[str, Any] = defaultdict(dict)
    for (method, task_name, score), by_seed in auc_by_method_task_score.items():
        across_seeds[method].setdefault(task_name, {})[score] = summarise_across_seeds(by_seed)

    paired_across: Dict[str, Any] = defaultdict(dict)
    reference = PRIMARY_COMPARISON[0]
    for method in {m for m, _ in grouped}:
        for task_name in tasks:
            ref_by_seed = auc_by_method_task_score.get((method, task_name, reference), {})
            if not ref_by_seed:
                continue
            comparisons = []
            for score in PRIMARY_COMPARISON[1:]:
                other_by_seed = auc_by_method_task_score.get((method, task_name, score), {})
                if not other_by_seed:
                    continue
                comparisons.append(
                    across_seed_comparison(
                        reference, score, ref_by_seed, other_by_seed
                    ).as_dict()
                )
            if comparisons:
                paired_across[method][task_name] = comparisons

    return {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inputs": [f.path for f in files],
            "legacy_inputs": [f.path for f in files if f.legacy],
            "scores_evaluated": scores,
            "scores_absent_from_inputs": absent,
            "tasks": tasks,
            "primary_task": primary_task,
            "orientations": table.as_dict(),
            "orientation_policy": (
                "AUCs use the preregistered direction for each (score, task). "
                "max(AUC, 1-AUC) is not used anywhere in this evaluation."
            ),
            "provenance": run_provenance(),
            "input_metadata": [
                {"path": f.path, "legacy": f.legacy, "metadata": f.metadata} for f in files
            ],
        },
        "per_run": per_run,
        "across_seeds": dict(across_seeds),
        "paired_within_run": paired_runs,
        "paired_across_seeds": dict(paired_across),
        "correlations": correlations,
    }


# ------------------------------------------------------------------- reporting


def jsonable(value: Any) -> Any:
    """Recursively replace non-finite floats with ``None``.

    ``json.dump`` writes ``NaN`` and ``Infinity`` literals, which Python reads
    back but which are not valid JSON and which strict parsers reject. In these
    results a NaN always means "not computed" -- a degenerate task, a constant
    input, an unresolvable operating point -- and ``null`` says that in a form
    every parser understands.

    Score *rows* are written by ``schema.write_score_file``, not here, and
    deliberately keep ``-Infinity``: it is a meaningful ranking position for
    ``combined_log_score`` and collapsing it to ``null`` would drop the sample.
    """
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except Exception:  # pragma: no cover - defensive
            pass
    return str(value)


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "  --  "
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return "  nan "
    return f"{value:.{digits}f}"


def format_report(results: Mapping[str, Any], table: OrientationTable) -> str:
    """A readable summary of the JSON results."""
    lines: List[str] = []
    metadata = results["metadata"]
    primary = metadata["primary_task"]

    lines.append("# Deletion-set inference: head-to-head evaluation")
    lines.append("")
    lines.append(f"Generated {metadata['timestamp']}")
    lines.append(f"Inputs: {len(metadata['inputs'])} score file(s)"
                 + (f", {len(metadata['legacy_inputs'])} legacy"
                    if metadata["legacy_inputs"] else ""))
    lines.append(f"Primary task: **{primary}**")
    lines.append("")
    lines.append("All AUCs use the preregistered orientation for each (score, task) "
                 "pair. `max(AUC, 1-AUC)` is not used. A score below 0.50 is "
                 "reported below 0.50.")
    lines.append("")

    if metadata["scores_absent_from_inputs"]:
        lines.append("## Scores absent from the inputs")
        lines.append("")
        for score in metadata["scores_absent_from_inputs"]:
            lines.append(f"- `{score}` -- not present in any input file")
        lines.append("")

    lines.append(f"## Per run, task `{primary}`")
    lines.append("")
    header = f"| {'run':<18} | {'score':<20} | {'dir':<4} | {'AUC':^8} | {'95% CI':^18} | {'AP':^8} | {'TPR@1%':^8} |"
    lines.append(header)
    lines.append("|" + "-" * 20 + "|" + "-" * 22 + "|" + "-" * 6 + "|" + "-" * 10
                 + "|" + "-" * 20 + "|" + "-" * 10 + "|" + "-" * 10 + "|")

    for run_key, run in sorted(results["per_run"].items()):
        task_block = run["tasks"].get(primary, {})
        for score, entry in sorted(task_block.get("scores", {}).items()):
            if not entry.get("available"):
                continue
            arrow = "up" if entry["direction"] == "higher_is_positive" else "down"
            tpr = entry["tpr_at_fpr"].get("0.01", {})
            lines.append(
                f"| {run_key:<18} | {score:<20} | {arrow:<4} | {_fmt(entry['auc']):^8} | "
                f"[{_fmt(entry['ci_lower'])}, {_fmt(entry['ci_upper'])}] | "
                f"{_fmt(entry['average_precision']):^8} | "
                f"{_fmt(tpr.get('tpr')):^8} |"
            )
    lines.append("")

    lines.append("## Across seeds (mean +/- SD of the per-seed AUC)")
    lines.append("")
    for method, tasks_block in sorted(results["across_seeds"].items()):
        lines.append(f"### {method}")
        lines.append("")
        lines.append(f"| {'task':<18} | {'score':<20} | {'mean':^8} | {'SD':^8} | {'n seeds':^8} |")
        lines.append("|" + "-" * 20 + "|" + "-" * 22 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 10 + "|")
        for task_name, score_block in sorted(tasks_block.items()):
            for score, summary in sorted(score_block.items()):
                lines.append(
                    f"| {task_name:<18} | {score:<20} | {_fmt(summary['mean']):^8} | "
                    f"{_fmt(summary['sd']):^8} | {summary['n_seeds']:^8} |"
                )
        lines.append("")

    lines.append("## Paired across-seed comparisons (unit = seed)")
    lines.append("")
    lines.append("Positive `mean diff` means the reference score has the higher AUC.")
    lines.append("")
    for method, tasks_block in sorted(results["paired_across_seeds"].items()):
        for task_name, comparisons in sorted(tasks_block.items()):
            if task_name != primary:
                continue
            lines.append(f"### {method}, task `{task_name}`")
            lines.append("")
            lines.append(f"| {'reference':<20} | {'vs':<20} | {'mean diff':^10} | {'SD':^8} | {'n':^4} | {'test':<16} | {'p':^8} |")
            lines.append("|" + "-" * 22 + "|" + "-" * 22 + "|" + "-" * 12 + "|" + "-" * 10
                         + "|" + "-" * 6 + "|" + "-" * 18 + "|" + "-" * 10 + "|")
            for comparison in comparisons:
                lines.append(
                    f"| {comparison['score_a']:<20} | {comparison['score_b']:<20} | "
                    f"{_fmt(comparison['mean_diff']):^10} | {_fmt(comparison['sd_diff']):^8} | "
                    f"{comparison['n_seeds']:^4} | {comparison['test']:<16} | "
                    f"{_fmt(comparison['p_value']):^8} |"
                )
            note = next((c.get("note") for c in comparisons if c.get("note")), "")
            if note:
                lines.append("")
                lines.append(f"> {note}")
            lines.append("")

    lines.append("## Diagnostic Spearman correlations")
    lines.append("")
    lines.append(f"| {'run':<18} | {'pair':<36} | {'rho':^8} | {'n':^6} |")
    lines.append("|" + "-" * 20 + "|" + "-" * 38 + "|" + "-" * 10 + "|" + "-" * 8 + "|")
    for run_key, block in sorted(results["correlations"].items()):
        for pair, entry in sorted(block.items()):
            lines.append(
                f"| {run_key:<18} | {pair:<36} | {_fmt(entry['rho']):^8} | {entry['n']:^6} |"
            )
    lines.append("")

    unresolvable = set()
    for run in results["per_run"].values():
        for task_block in run["tasks"].values():
            for entry in task_block.get("scores", {}).values():
                if not entry.get("available"):
                    continue
                for target, tpr in entry["tpr_at_fpr"].items():
                    if not tpr["resolvable"] and tpr.get("note"):
                        unresolvable.add((target, tpr["note"]))
    if unresolvable:
        lines.append("## Suppressed operating points")
        lines.append("")
        for target, note in sorted(unresolvable):
            lines.append(f"- FPR {target}: {note}")
        lines.append("")

    return "\n".join(lines)


# ------------------------------------------------------------------------- CLI


def expand_inputs(patterns: Sequence[str]) -> List[str]:
    """Expand glob patterns into an ordered, de-duplicated file list."""
    paths: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches and os.path.exists(pattern):
            matches = [pattern]
        if not matches:
            raise FileNotFoundError(f"No file matched {pattern!r}")
        for match in matches:
            if match not in paths:
                paths.append(match)
    return paths


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute deletion-set inference metrics from saved score tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # new-format tables\n"
            "  $PY -m deletion_signals.offline_eval \\\n"
            "      --inputs 'ds_results/seed*/**/per_sample_scores.json' \\\n"
            "      --output ds_results/evaluation\n\n"
            "  # legacy per_sample_components.json from the sibling pipeline\n"
            "  $PY -m deletion_signals.offline_eval --legacy \\\n"
            "      --inputs 'unlearning_paradox_sanity/results_seed4*/*/per_sample_components.json' \\\n"
            "      --output ds_results/legacy_evaluation\n"
        ),
    )
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Score files or glob patterns")
    parser.add_argument("--output", required=True,
                        help="Output directory for results.json and report.md")
    parser.add_argument("--config", default=None, help="YAML config (evaluation block)")
    parser.add_argument("--legacy", action="store_true",
                        help="Inputs are legacy per_sample_components.json files")
    parser.add_argument("--legacy-seed", type=int, default=None,
                        help="Seed for legacy files whose path does not encode one")
    parser.add_argument("--legacy-dataset", default="unknown")
    parser.add_argument("--legacy-model", default="unknown")
    parser.add_argument("--orientations", default=None,
                        help="Override the orientation table path")
    parser.add_argument("--n-bootstrap", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.n_bootstrap is not None:
        config["evaluation"]["n_bootstrap"] = args.n_bootstrap
    if args.orientations:
        config["evaluation"]["orientations_path"] = args.orientations

    table = OrientationTable.load(config["evaluation"].get("orientations_path"))

    legacy_kwargs: Dict[str, Any] = {}
    if args.legacy:
        legacy_kwargs = {
            "seed": args.legacy_seed,
            "dataset": args.legacy_dataset,
            "model_id": args.legacy_model,
        }

    files = [load_any(path, **legacy_kwargs) for path in expand_inputs(args.inputs)]
    if any(f.legacy for f in files) and not args.quiet:
        print(
            "NOTE: legacy inputs are in degraded mode -- they carry only the "
            "alignment, gradient-magnitude and self-influence columns. Every "
            "other score is reported as absent, not as zero."
        )

    results = evaluate(files, config, table)

    os.makedirs(args.output, exist_ok=True)
    results_path = os.path.join(args.output, "results.json")
    with open(results_path, "w") as handle:
        json.dump(jsonable(results), handle, indent=2, allow_nan=False)

    report = format_report(results, table)
    report_path = os.path.join(args.output, "report.md")
    with open(report_path, "w") as handle:
        handle.write(report)

    if not args.quiet:
        print(report)
    print(f"\nWrote {results_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
