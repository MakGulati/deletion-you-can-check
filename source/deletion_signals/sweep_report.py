"""Unlearning-strength sweep: utility, forgetting, membership and deletion.

The offline evaluator answers "which score separates the populations best".
This module answers the question the sweep exists for, which is different:

    as unlearning strengthens and the membership evidence disappears, does the
    signed cosine still identify the deletion set while retained-data utility
    survives?

That needs three quantities the evaluator does not compute -- how much utility
the model lost, how much forgetting actually happened, and whether membership
evidence is still present -- read at every strength, so that a deletion result
can be placed against them rather than reported on its own.

The decision rule and its thresholds are fixed in advance in
``docs/PREREGISTRATION.md`` and appear here as module constants. Nothing in this
module chooses a threshold, a direction or a winner from the data: it applies
the rule and prints which criteria each setting met.

Unit of replication
-------------------
The **seed** -- an independently fine-tuned and unlearned model. Every headline
number is a mean over seeds with the across-seed SD beside it, and a setting
qualifies only if every one of its seeds qualifies. Per-sample bootstrap
intervals appear only where the preregistered rule names one; they describe the
precision of one run's AUC and cannot support a claim about the method.

Usage
-----
::

    $PY -m deletion_signals.sweep_report \\
        --inputs 'ds_sweep/ascent_*/seed*/ga_gd/per_sample_scores.json' \\
        --output ds_sweep/analysis
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .orientations import OrientationTable, build_task_arrays
from .reproducibility import new_rng, run_provenance
from .schema import POPULATIONS, SCORE_COLUMNS, ScoreFile, load_any
from .stats import bootstrap_auc_ci, roc_auc, summarise_across_seeds

# --------------------------------------------------------------------- the rule
#
# Preregistered in docs/PREREGISTRATION.md before the sweep ran. These are the
# thresholds, not defaults to be tuned: changing one changes the verdict, so a
# change belongs in the document and in the git history, not in a flag.

MEMBERSHIP_SCORE = "loss_ratio"
MEMBERSHIP_TASK = "forget_vs_unseen"      # forget vs never-trained data
MEMBERSHIP_GONE_BAND = (0.45, 0.55)

UTILITY_MAX_PPL_RATIO = 1.10              # unlearned / fine-tuned, per population
UTILITY_POPULATIONS = ("retain", "test")

DELETION_SCORE = "raw_cosine"
DELETION_TASK = "forget_vs_retain"
DELETION_MIN_AUC = 0.70
DELETION_CI_LOWER_ABOVE = 0.50

CRITERIA = ("membership_gone", "utility_preserved", "deletion_present")

# NOT a criterion: the forget/retain loss ratio that GA+GD's own early-stopping
# rule targets, and that no baseline seed reached. Drawn on the forgetting panel
# as the reference the sweep is trying to get past, and reported in the tables.
FORGET_RATIO_TARGET = 2.0

# Tasks reported at every strength. The membership pair is present so a deletion
# result can never be read without the membership context beside it.
REPORT_TASKS = (
    "forget_vs_retain",
    "forget_vs_unseen",
    "member_vs_unseen",
    "retain_vs_unseen",
)

# Baselines the primary score is shown against, in the order they appear.
DELETION_BASELINES = (
    "alignment_cos_sq",
    "loss_ratio_vs_finetuned",
    "loss_delta_vs_finetuned",
    "loss_ratio",
    "grad_ratio",
    "grad_norm_sq",
    "fuma_lora_ratio",
    "sif",
    "combined_log_score",
    "loss_unlearned",
)

# Which config key holds "unlearning strength" for each method.
_STRENGTH_KEY_BY_METHOD = {"ga_gd": "ascent_lr", "scrub": "learning_rate"}


# ------------------------------------------------------------------ aggregation


def token_weighted_mean_nll(
    losses: Sequence[Optional[float]], n_tokens: Sequence[Optional[int]]
) -> Tuple[Optional[float], bool]:
    """Mean NLL per token over a population, and whether weights were available.

    Each row's ``loss`` is already a mean over that sample's tokens, so a plain
    average over rows weights a six-token sample the same as a five-hundred-token
    one and the resulting "perplexity" is not the perplexity of any corpus. The
    token-weighted mean, sum(loss_i * n_i) / sum(n_i), is.

    Falls back to the unweighted mean when token counts are missing (legacy
    tables carry none) and returns ``weighted=False`` so the report can say so
    rather than quietly presenting a different quantity under the same name.
    """
    pairs = [
        (float(loss), n)
        for loss, n in zip(losses, n_tokens)
        if loss is not None and math.isfinite(float(loss))
    ]
    if not pairs:
        return None, False

    usable = [(loss, int(n)) for loss, n in pairs if n is not None and int(n) > 0]
    if len(usable) == len(pairs) and usable:
        total_tokens = sum(n for _, n in usable)
        return sum(loss * n for loss, n in usable) / total_tokens, True

    return sum(loss for loss, _ in pairs) / len(pairs), False


def perplexity(mean_nll: Optional[float]) -> Optional[float]:
    """exp(mean token NLL), or None. Overflow is reported as infinite, not hidden."""
    if mean_nll is None or not math.isfinite(mean_nll):
        return None
    try:
        return math.exp(mean_nll)
    except OverflowError:
        return float("inf")


def ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None:
        return None
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0:
        return None
    return numerator / denominator


def run_strength(metadata: Mapping[str, Any], method: str) -> Tuple[float, str]:
    """The swept hyperparameter for one run, and where it was read from.

    Read from the run's own recorded hyperparameters rather than from its
    directory name: a path is a label someone typed, and a sweep that mislabels
    one cell would produce a monotone-looking curve out of a transposition.
    """
    key = _STRENGTH_KEY_BY_METHOD.get(method, "learning_rate")

    hyperparameters = (metadata.get("unlearning_run") or {}).get("hyperparameters") or {}
    if hyperparameters.get(key) is not None:
        return float(hyperparameters[key]), f"unlearning_run.hyperparameters.{key}"

    effective = (
        (metadata.get("provenance") or {}).get("effective_config") or {}
    ).get("unlearning") or {}
    value = (effective.get(method) or {}).get(key)
    if value is not None:
        return float(value), f"provenance.effective_config.unlearning.{method}.{key}"

    raise ValueError(
        f"No {key!r} recorded for method {method!r} in this score file's metadata "
        f"(looked in unlearning_run.hyperparameters and in the effective config). "
        f"A sweep point with no strength is unplaceable; re-run the cell so the "
        f"unlearning record is written, or evaluate it with offline_eval instead."
    )


@dataclass
class CellRun:
    """One score file: one strength, one seed, one method."""

    path: str
    seed: int
    method: str
    strength: float
    strength_source: str
    counts: Dict[str, int] = field(default_factory=dict)
    token_weighted: bool = True
    losses: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)
    perplexities: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)
    forget_retain_loss_ratio: Optional[float] = None
    aucs: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    unlearning: Dict[str, Any] = field(default_factory=dict)
    criteria: Dict[str, Any] = field(default_factory=dict)

    def auc(self, task: str, score: str) -> Optional[float]:
        entry = self.aucs.get(task, {}).get(score)
        return None if entry is None else entry.get("auc")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "seed": self.seed,
            "method": self.method,
            "strength": self.strength,
            "strength_source": self.strength_source,
            "counts": self.counts,
            "token_weighted_perplexity": self.token_weighted,
            "mean_token_nll": self.losses,
            "perplexity": self.perplexities,
            "forget_retain_loss_ratio": self.forget_retain_loss_ratio,
            "auc": self.aucs,
            "unlearning": self.unlearning,
            "criteria": self.criteria,
        }


def summarise_populations(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Per-population mean NLL and perplexity under both models."""
    losses: Dict[str, Dict[str, Optional[float]]] = {}
    perplexities: Dict[str, Dict[str, Optional[float]]] = {}
    counts: Dict[str, int] = {}
    all_weighted = True

    for population in POPULATIONS:
        subset = [r for r in rows if str(r.get("population")) == population]
        counts[population] = len(subset)
        n_tokens = [r.get("n_tokens") for r in subset]

        entry: Dict[str, Optional[float]] = {}
        for model_key, column in (("unlearned", "loss_unlearned"),
                                  ("finetuned", "loss_finetuned")):
            mean_nll, weighted = token_weighted_mean_nll(
                [r.get(column) for r in subset], n_tokens
            )
            if mean_nll is not None:
                all_weighted = all_weighted and weighted
            entry[model_key] = mean_nll
        losses[population] = entry
        perplexities[population] = {
            "unlearned": perplexity(entry["unlearned"]),
            "finetuned": perplexity(entry["finetuned"]),
            "ratio": ratio(
                perplexity(entry["unlearned"]), perplexity(entry["finetuned"])
            ),
        }

    return {
        "losses": losses,
        "perplexities": perplexities,
        "counts": counts,
        "token_weighted": all_weighted,
    }


def evaluate_cell(
    score_file: ScoreFile,
    table: OrientationTable,
    n_bootstrap: int,
    confidence: float,
) -> CellRun:
    """Every quantity the decision rule needs, for one score file."""
    rows = score_file.rows
    if not rows:
        raise ValueError(f"{score_file.path} contains no rows")

    metadata = score_file.metadata
    method = str(metadata.get("method") or rows[0].get("method", "unknown"))
    seed = int(metadata.get("seed", rows[0].get("seed", -1)))
    strength, strength_source = run_strength(metadata, method)

    populations = summarise_populations(rows)

    cell = CellRun(
        path=score_file.path or "",
        seed=seed,
        method=method,
        strength=strength,
        strength_source=strength_source,
        counts=populations["counts"],
        token_weighted=populations["token_weighted"],
        losses=populations["losses"],
        perplexities=populations["perplexities"],
        forget_retain_loss_ratio=ratio(
            populations["losses"]["forget"]["unlearned"],
            populations["losses"]["retain"]["unlearned"],
        ),
    )

    unlearning_run = metadata.get("unlearning_run") or {}
    cell.unlearning = {
        "steps_configured": unlearning_run.get("steps_configured"),
        "steps_taken": unlearning_run.get("steps_taken"),
        "stop_reason": unlearning_run.get("stop_reason"),
        "final_forget_loss": unlearning_run.get("final_forget_loss"),
        "final_retain_loss": unlearning_run.get("final_retain_loss"),
    }

    available = set(score_file.available_scores)
    # A bootstrap interval is computed only where the preregistered rule names
    # one. Everywhere else the across-seed SD is the uncertainty that matters,
    # and a per-sample interval beside it would invite the wrong reading.
    needs_ci = {(DELETION_TASK, DELETION_SCORE), (MEMBERSHIP_TASK, MEMBERSHIP_SCORE)}

    for task_name in REPORT_TASKS:
        task = table.task(task_name)
        block: Dict[str, Dict[str, Any]] = {}
        for score in SCORE_COLUMNS:
            if score not in available:
                continue
            values, labels, dropped = build_task_arrays(rows, score, task)
            if not values or len(set(labels)) < 2:
                continue
            oriented = table.orient(values, score, task_name)
            entry: Dict[str, Any] = {
                "direction": table.orientation(score, task_name).direction,
                "n_dropped": dropped,
            }
            if (task_name, score) in needs_ci:
                estimate = bootstrap_auc_ci(
                    oriented,
                    labels,
                    n_bootstrap=n_bootstrap,
                    confidence=confidence,
                    rng=new_rng(seed),
                )
                entry.update(estimate.as_dict())
            else:
                entry["auc"] = roc_auc(oriented, labels)
                entry["n_positive"] = int(sum(labels))
                entry["n_negative"] = int(len(labels) - sum(labels))
            block[score] = entry
        cell.aucs[task_name] = block

    cell.criteria = apply_decision_rule(cell)
    return cell


def apply_decision_rule(cell: CellRun) -> Dict[str, Any]:
    """The preregistered three-part rule, applied to one run.

    Each criterion carries the numbers it was decided from, so a reader can
    check the verdict without re-deriving it, and each is reported separately:
    "genuinely forgotten" is never concluded from one of them.
    """
    membership_entry = cell.aucs.get(MEMBERSHIP_TASK, {}).get(MEMBERSHIP_SCORE, {})
    membership_auc = membership_entry.get("auc")
    low, high = MEMBERSHIP_GONE_BAND
    membership_gone = (
        membership_auc is not None
        and math.isfinite(membership_auc)
        and low <= membership_auc <= high
    )

    utility_ratios = {
        population: cell.perplexities.get(population, {}).get("ratio")
        for population in UTILITY_POPULATIONS
    }
    known = [r for r in utility_ratios.values() if r is not None and math.isfinite(r)]
    utility_preserved = (
        len(known) == len(UTILITY_POPULATIONS)
        and all(r <= UTILITY_MAX_PPL_RATIO for r in known)
    )

    deletion_entry = cell.aucs.get(DELETION_TASK, {}).get(DELETION_SCORE, {})
    deletion_auc = deletion_entry.get("auc")
    deletion_ci_lower = deletion_entry.get("ci_lower")
    deletion_present = (
        deletion_auc is not None
        and math.isfinite(deletion_auc)
        and deletion_auc >= DELETION_MIN_AUC
        and deletion_ci_lower is not None
        and math.isfinite(deletion_ci_lower)
        and deletion_ci_lower > DELETION_CI_LOWER_ABOVE
    )

    return {
        "membership_gone": {
            "met": bool(membership_gone),
            "score": MEMBERSHIP_SCORE,
            "task": MEMBERSHIP_TASK,
            "auc": membership_auc,
            "band": list(MEMBERSHIP_GONE_BAND),
        },
        "utility_preserved": {
            "met": bool(utility_preserved),
            "ppl_ratio": utility_ratios,
            "max_allowed": UTILITY_MAX_PPL_RATIO,
            "evaluable": len(known) == len(UTILITY_POPULATIONS),
        },
        "deletion_present": {
            "met": bool(deletion_present),
            "score": DELETION_SCORE,
            "task": DELETION_TASK,
            "auc": deletion_auc,
            "ci_lower": deletion_ci_lower,
            "min_auc": DELETION_MIN_AUC,
            "ci_lower_above": DELETION_CI_LOWER_ABOVE,
        },
        "qualifies": bool(membership_gone and utility_preserved and deletion_present),
    }


# ----------------------------------------------------------------- across seeds


def _by_seed(cells: Sequence[CellRun], getter) -> Dict[int, float]:
    values: Dict[int, float] = {}
    for cell in cells:
        value = getter(cell)
        if value is not None:
            try:
                values[cell.seed] = float(value)
            except (TypeError, ValueError):
                continue
    return values


def summarise_setting(cells: Sequence[CellRun]) -> Dict[str, Any]:
    """Mean and across-seed SD of every reported quantity at one strength."""
    summary: Dict[str, Any] = {
        "strength": cells[0].strength,
        "method": cells[0].method,
        "seeds": sorted(cell.seed for cell in cells),
        "n_seeds": len(cells),
    }

    summary["perplexity_ratio"] = {
        population: summarise_across_seeds(
            _by_seed(cells, lambda c, p=population: c.perplexities.get(p, {}).get("ratio"))
        )
        for population in POPULATIONS
    }
    summary["perplexity_unlearned"] = {
        population: summarise_across_seeds(
            _by_seed(
                cells, lambda c, p=population: c.perplexities.get(p, {}).get("unlearned")
            )
        )
        for population in POPULATIONS
    }
    summary["forget_retain_loss_ratio"] = summarise_across_seeds(
        _by_seed(cells, lambda c: c.forget_retain_loss_ratio)
    )
    summary["final_forget_loss"] = summarise_across_seeds(
        _by_seed(cells, lambda c: c.unlearning.get("final_forget_loss"))
    )
    summary["stop_reasons"] = sorted(
        {str(cell.unlearning.get("stop_reason")) for cell in cells}
    )

    auc_summary: Dict[str, Dict[str, Any]] = {}
    for task_name in REPORT_TASKS:
        scores = sorted({s for cell in cells for s in cell.aucs.get(task_name, {})})
        auc_summary[task_name] = {
            score: summarise_across_seeds(
                _by_seed(cells, lambda c, t=task_name, s=score: c.auc(t, s))
            )
            for score in scores
        }
    summary["auc"] = auc_summary

    # Paired differences on the primary task: the same seed's primary score
    # minus the same seed's baseline. Paired, so the between-seed variation that
    # both share cancels -- which is why this and not two separate means.
    paired: List[Dict[str, Any]] = []
    for baseline in DELETION_BASELINES:
        differences = _by_seed(
            cells,
            lambda c, b=baseline: (
                None
                if c.auc(DELETION_TASK, DELETION_SCORE) is None
                or c.auc(DELETION_TASK, b) is None
                else c.auc(DELETION_TASK, DELETION_SCORE) - c.auc(DELETION_TASK, b)
            ),
        )
        if differences:
            paired.append(
                {
                    "score_a": DELETION_SCORE,
                    "score_b": baseline,
                    "task": DELETION_TASK,
                    **summarise_across_seeds(differences),
                }
            )
    summary["paired_differences"] = paired

    per_criterion = {
        criterion: sorted(
            cell.seed for cell in cells if cell.criteria[criterion]["met"]
        )
        for criterion in CRITERIA
    }
    summary["criteria"] = {
        "seeds_meeting": per_criterion,
        "n_seeds_meeting": {k: len(v) for k, v in per_criterion.items()},
        # A setting qualifies only if EVERY seed qualifies. Two out of three is
        # reported as two out of three, never rounded up to "qualifies".
        "qualifies_on_all_seeds": all(
            len(per_criterion[criterion]) == len(cells) for criterion in CRITERIA
        ),
    }
    return summary


def build_results(
    files: Sequence[ScoreFile],
    table: OrientationTable,
    n_bootstrap: int,
    confidence: float,
) -> Dict[str, Any]:
    """Evaluate every file, group by (method, strength), and summarise."""
    cells = [evaluate_cell(f, table, n_bootstrap, confidence) for f in files]

    grouped: Dict[Tuple[str, float], List[CellRun]] = defaultdict(list)
    for cell in cells:
        grouped[(cell.method, cell.strength)].append(cell)

    duplicates = {
        key: [c.path for c in group]
        for key, group in grouped.items()
        if len({c.seed for c in group}) != len(group)
    }
    if duplicates:
        raise ValueError(
            f"Two score files share a (method, strength, seed): {duplicates}. One "
            f"would silently overwrite the other in the across-seed summary."
        )

    settings = [
        summarise_setting(sorted(group, key=lambda c: c.seed))
        for _, group in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    ]

    qualifying = [s for s in settings if s["criteria"]["qualifies_on_all_seeds"]]

    return {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inputs": [f.path for f in files],
            "n_bootstrap": n_bootstrap,
            "confidence": confidence,
            "replication_unit": "seed (independently fine-tuned and unlearned model)",
            "preregistration": {
                "document": "docs/PREREGISTRATION.md",
                "membership_gone": {
                    "score": MEMBERSHIP_SCORE,
                    "task": MEMBERSHIP_TASK,
                    "band": list(MEMBERSHIP_GONE_BAND),
                },
                "utility_preserved": {
                    "populations": list(UTILITY_POPULATIONS),
                    "max_perplexity_ratio": UTILITY_MAX_PPL_RATIO,
                },
                "deletion_present": {
                    "score": DELETION_SCORE,
                    "task": DELETION_TASK,
                    "min_auc": DELETION_MIN_AUC,
                    "ci_lower_above": DELETION_CI_LOWER_ABOVE,
                },
                "all_criteria_on_all_seeds": True,
            },
            "orientations": table.as_dict(),
            "provenance": run_provenance(),
        },
        "runs": [cell.as_dict() for cell in sorted(cells, key=lambda c: (c.method, c.strength, c.seed))],
        "settings": settings,
        "verdict": {
            "qualifying_settings": [
                {"method": s["method"], "strength": s["strength"]} for s in qualifying
            ],
            "outcome": _outcome(settings, qualifying),
        },
    }


def _outcome(settings: Sequence[Mapping[str, Any]], qualifying: Sequence[Mapping[str, Any]]) -> str:
    """Which of the three preregistered outcomes the sweep produced.

    All three are reportable results. The wording is fixed here so the verdict
    is mechanical rather than a judgement made while looking at the numbers.
    """
    if qualifying:
        return (
            "fingerprint persists: at least one strength satisfies all three "
            "criteria on every seed -- membership evidence is gone, retained "
            "utility survives, and the deletion signal is still present"
        )

    membership_gone_anywhere = any(
        s["criteria"]["n_seeds_meeting"]["membership_gone"] == s["n_seeds"]
        for s in settings
    )
    utility_anywhere = any(
        s["criteria"]["n_seeds_meeting"]["utility_preserved"] == s["n_seeds"]
        for s in settings
    )
    deletion_anywhere = any(
        s["criteria"]["n_seeds_meeting"]["deletion_present"] == s["n_seeds"]
        for s in settings
    )

    if membership_gone_anywhere and utility_anywhere and not deletion_anywhere:
        return (
            "incomplete forgetting: where membership evidence disappears and "
            "utility survives, the deletion signal does not clear its threshold "
            "-- the fingerprint in the baseline runs is consistent with "
            "unlearning simply not having finished"
        )
    if not (membership_gone_anywhere and utility_anywhere):
        return (
            "inconclusive: no strength achieved forgetting and utility together, "
            "so the sweep cannot say what happens to the deletion signal once "
            "membership evidence is gone. This is a statement about the range "
            "swept, not about the signal"
        )
    return (
        "inconclusive: the three criteria are met by different settings but by "
        "no single one on every seed"
    )


# ------------------------------------------------------------------- reporting


def _fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "--"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    # A collapsed run produces a perplexity ratio of order 1e7, and printing
    # that to three decimals is thirteen characters of noise around one fact:
    # the model is gone.
    if value != 0.0 and (abs(value) >= 1e5 or abs(value) < 1e-4):
        return f"{value:.{max(digits - 1, 1)}e}"
    return f"{value:.{digits}f}"


def _fmt_mean_sd(summary: Optional[Mapping[str, Any]], digits: int = 3) -> str:
    """``mean ± SD`` over seeds, with the SD absent at one seed rather than 0."""
    if not summary or not summary.get("n_seeds"):
        return "--"
    sd = summary.get("sd")
    if sd is None or (isinstance(sd, float) and math.isnan(sd)):
        return f"{_fmt(summary['mean'], digits)} (n=1)"
    return f"{_fmt(summary['mean'], digits)} ± {_fmt(sd, digits)}"


def _mark(met: bool) -> str:
    return "yes" if met else "no"


def format_report(results: Mapping[str, Any]) -> str:
    """The sweep report, in the order a reader needs it."""
    lines: List[str] = []
    metadata = results["metadata"]
    settings = results["settings"]
    prereg = metadata["preregistration"]

    lines += [
        "# GA+GD unlearning-strength sweep",
        "",
        f"Generated {metadata['timestamp']} from {len(metadata['inputs'])} score file(s).",
        "",
        "## The question, and why the baseline runs could not answer it",
        "",
        "On `forget_vs_unseen` the calibrated loss ratio separates forgotten samples",
        "from never-trained ones almost perfectly -- in the baseline runs, AUC 0.001",
        "under the preregistered higher-is-forget direction, i.e. 0.999 in the",
        "membership direction, because a forgotten sample still carries the lower",
        "loss of a fitted member. On that task the metric measures **membership**,",
        "not deletion. On `forget_vs_retain` both classes are members and membership",
        "cannot explain a result -- but in the baseline runs no seed reached its",
        "forget/retain loss-ratio target, so a directional signal there is also",
        "consistent with unlearning simply not having finished.",
        "",
        "This sweep varies the ascent learning rate with everything else held fixed",
        "and asks: **as unlearning strengthens and the membership evidence",
        "disappears, does the signed cosine still identify the deletion set while",
        "retained-data utility survives?**",
        "",
        "## The preregistered decision rule",
        "",
        f"Fixed in `{prereg['document']}` before the sweep ran.",
        "",
        "| criterion | test |",
        "|---|---|",
        (
            f"| membership gone | `{prereg['membership_gone']['score']}` AUC on "
            f"`{prereg['membership_gone']['task']}` within "
            f"[{prereg['membership_gone']['band'][0]}, {prereg['membership_gone']['band'][1]}] |"
        ),
        (
            f"| utility preserved | {' and '.join(prereg['utility_preserved']['populations'])} "
            f"perplexity under the unlearned model <= "
            f"{prereg['utility_preserved']['max_perplexity_ratio']} x the fine-tuned model's |"
        ),
        (
            f"| deletion present | `{prereg['deletion_present']['score']}` AUC on "
            f"`{prereg['deletion_present']['task']}` >= {prereg['deletion_present']['min_auc']} "
            f"AND bootstrap CI lower bound > {prereg['deletion_present']['ci_lower_above']} |"
        ),
        "",
        "A setting qualifies only if **all three hold on all seeds**. Each criterion",
        "is reported separately; \"genuinely forgotten\" is never concluded from one.",
        "",
        "## Verdict",
        "",
        f"**{results['verdict']['outcome']}**",
        "",
    ]

    qualifying = results["verdict"]["qualifying_settings"]
    if qualifying:
        listed = ", ".join(f"{q['method']} @ {q['strength']:g}" for q in qualifying)
        lines += [f"Qualifying setting(s): {listed}.", ""]
    else:
        lines += ["No setting qualified on every seed.", ""]

    lines += [
        "| method | strength | membership gone | utility preserved | deletion present | qualifies |",
        "|---|---:|---|---|---|---|",
    ]
    for setting in settings:
        met = setting["criteria"]["n_seeds_meeting"]
        n = setting["n_seeds"]
        lines.append(
            f"| {setting['method']} | {setting['strength']:g} | "
            f"{met['membership_gone']}/{n} | {met['utility_preserved']}/{n} | "
            f"{met['deletion_present']}/{n} | "
            f"{_mark(setting['criteria']['qualifies_on_all_seeds'])} |"
        )
    lines.append("")

    # ---- what the unlearning run actually did ------------------------------
    lines += [
        "## What each unlearning run actually did",
        "",
        "`stop_reason` matters: a run that tripped the collapse guard is a different",
        "experiment from one that ran to completion, and it is the \"utility collapses",
        "before membership disappears\" branch of the rule, not a failed run.",
        "",
        "| method | strength | steps taken | stop reason(s) | final forget loss | forget/retain loss ratio |",
        "|---|---:|---|---|---|---|",
    ]
    for setting in settings:
        steps = sorted({
            str(run["unlearning"].get("steps_taken"))
            for run in results["runs"]
            if run["method"] == setting["method"] and run["strength"] == setting["strength"]
        })
        lines.append(
            f"| {setting['method']} | {setting['strength']:g} | {', '.join(steps)} | "
            f"{', '.join(setting['stop_reasons'])} | "
            f"{_fmt_mean_sd(setting['final_forget_loss'])} | "
            f"{_fmt_mean_sd(setting['forget_retain_loss_ratio'])} |"
        )
    lines.append("")

    # ---- utility ------------------------------------------------------------
    token_note = all(run["token_weighted_perplexity"] for run in results["runs"])
    lines += [
        "## Utility: perplexity under the unlearned model, relative to the fine-tuned one",
        "",
        (
            "Token-weighted: sum(loss_i x n_tokens_i) / sum(n_tokens_i), then exp."
            if token_note
            else "**Unweighted mean over samples** -- at least one input carries no "
            "token counts, so this is not a corpus perplexity. Reported as measured."
        ),
        "",
        "| method | strength | retain ppl ratio | test ppl ratio | forget ppl ratio |",
        "|---|---:|---|---|---|",
    ]
    for setting in settings:
        lines.append(
            f"| {setting['method']} | {setting['strength']:g} | "
            f"{_fmt_mean_sd(setting['perplexity_ratio']['retain'])} | "
            f"{_fmt_mean_sd(setting['perplexity_ratio']['test'])} | "
            f"{_fmt_mean_sd(setting['perplexity_ratio']['forget'])} |"
        )
    lines += [
        "",
        f"Threshold: retain and test both <= {UTILITY_MAX_PPL_RATIO}. The forget "
        "column is not a criterion -- it is the intended effect.",
        "",
    ]

    # ---- membership ---------------------------------------------------------
    lines += [
        "## Membership evidence",
        "",
        f"`{MEMBERSHIP_SCORE}` AUC. `{MEMBERSHIP_TASK}` is the criterion; the other",
        "two are shown beside it so a deletion result is never read without the",
        "membership context. All three use their preregistered directions.",
        "",
        "| method | strength | forget_vs_unseen | member_vs_unseen | retain_vs_unseen |",
        "|---|---:|---|---|---|",
    ]
    for setting in settings:
        aucs = setting["auc"]
        lines.append(
            f"| {setting['method']} | {setting['strength']:g} | "
            f"{_fmt_mean_sd(aucs.get('forget_vs_unseen', {}).get(MEMBERSHIP_SCORE))} | "
            f"{_fmt_mean_sd(aucs.get('member_vs_unseen', {}).get(MEMBERSHIP_SCORE))} | "
            f"{_fmt_mean_sd(aucs.get('retain_vs_unseen', {}).get(MEMBERSHIP_SCORE))} |"
        )
    lines += [
        "",
        f"Membership counts as gone when the first column lies in "
        f"[{MEMBERSHIP_GONE_BAND[0]}, {MEMBERSHIP_GONE_BAND[1]}].",
        "",
    ]

    # ---- deletion -----------------------------------------------------------
    present_baselines = [
        b
        for b in DELETION_BASELINES
        if any(b in s["auc"].get(DELETION_TASK, {}) for s in settings)
    ]
    lines += [
        "## Deletion signal on `forget_vs_retain`",
        "",
        "Both classes are known training members here, so no membership argument can",
        "explain a result. Mean +/- across-seed SD.",
        "",
        "| method | strength | "
        + f"**{DELETION_SCORE}** | "
        + " | ".join(present_baselines)
        + " |",
        "|---|---:|" + "---|" * (1 + len(present_baselines)),
    ]
    for setting in settings:
        block = setting["auc"].get(DELETION_TASK, {})
        cells = [_fmt_mean_sd(block.get(DELETION_SCORE))] + [
            _fmt_mean_sd(block.get(b)) for b in present_baselines
        ]
        lines.append(
            f"| {setting['method']} | {setting['strength']:g} | " + " | ".join(cells) + " |"
        )
    lines.append("")

    lines += [
        "### Paired differences, primary minus baseline (unit = seed)",
        "",
        "Paired within seed, so the between-seed variation both scores share",
        "cancels. Positive means the primary score has the higher AUC. With n = 3",
        "seeds these differences and their SDs are the evidence; no p-value is",
        "computed here, and none would carry weight at this n.",
        "",
        "| method | strength | baseline | mean diff | SD | n seeds |",
        "|---|---:|---|---|---|---:|",
    ]
    for setting in settings:
        for comparison in setting["paired_differences"]:
            lines.append(
                f"| {setting['method']} | {setting['strength']:g} | "
                f"{comparison['score_b']} | {_fmt(comparison['mean'])} | "
                f"{_fmt(comparison['sd'])} | {comparison['n_seeds']} |"
            )
    lines.append("")

    # ---- threat model and caveats ------------------------------------------
    lines += [
        "## Threat model",
        "",
        "`raw_cosine` requires **both** the fine-tuned pre-unlearning checkpoint and",
        "the unlearned checkpoint, because ĝ = θ_unlearned − θ_finetuned cannot be",
        "formed without them. This is a **white-box deletion audit** -- something a",
        "model owner, a regulator or an authorised verifier can run -- and not a",
        "black-box membership-inference attack. A party holding only the released",
        "model cannot mount it.",
        "",
        "The claim is scoped to gradient-ascent-based unlearning. The mechanism",
        "argued for the sign of the cosine is specific to GA+GD: forget samples",
        "receive ascent and retain samples descent, so their gradients sit on",
        "opposite sides of ĝ. SCRUB showed no directional advantage and is out of",
        "this sweep.",
        "",
        "## Caveats that apply to every number above",
        "",
        "- **n = 3 seeds.** Means, paired differences and across-seed SDs lead. No",
        "  p-value is reported here; at three seeds none would be informative.",
        "- **`raw_cosine` was promoted to the headline after being seen to win** on",
        "  the baseline three seeds. It is preregistered for this sweep, so the sweep",
        "  tests it on data that did not select it -- but the baseline seeds cannot",
        "  serve as that confirmation, and nothing here should be read as if they do.",
        "- Bootstrap intervals are over samples within one run. They say how",
        "  precisely one model's AUC is estimated, not how it varies between models.",
        "",
    ]

    return "\n".join(lines)


# ------------------------------------------------------------------------ plot


# Four hues from the validated categorical order (blue, orange, aqua, yellow),
# assigned in fixed order and never cycled. Identity is carried by marker shape
# as well as hue, so the two low-contrast slots stay readable and the figure
# survives greyscale printing.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SERIES_MARKERS = ("o", "s", "^", "D")
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
SURFACE = "#fcfcfb"
THRESHOLD_COLOR = "#e34948"


def _series(axis, x, summaries, label, index, log_y=False):
    """One line with across-seed SD error bars, in the fixed slot ``index``."""
    means = [s["mean"] if s and s.get("n_seeds") else float("nan") for s in summaries]
    sds = [
        (s.get("sd") if s and s.get("n_seeds", 0) > 1 else 0.0) or 0.0
        for s in summaries
    ]
    sds = [0.0 if (sd is None or math.isnan(sd)) else sd for sd in sds]
    axis.errorbar(
        x,
        means,
        yerr=sds,
        label=label,
        color=SERIES_COLORS[index % len(SERIES_COLORS)],
        marker=SERIES_MARKERS[index % len(SERIES_MARKERS)],
        markersize=6,
        linewidth=2.0,
        capsize=3,
        elinewidth=1.2,
        zorder=3 - index * 0.1,
    )
    if log_y:
        axis.set_yscale("log")


def _style(axis, title, ylabel, xlabel="ascent learning rate"):
    from matplotlib.ticker import ScalarFormatter

    axis.set_title(title, fontsize=11, color=INK_PRIMARY, loc="left", pad=8)
    axis.set_ylabel(ylabel, fontsize=9, color=INK_SECONDARY)
    axis.set_xscale("log")
    axis.set_xlabel(xlabel, fontsize=9, color=INK_SECONDARY)
    axis.grid(True, which="major", color=INK_MUTED, alpha=0.25, linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(INK_MUTED)
    axis.tick_params(colors=INK_SECONDARY, labelsize=8)
    if axis.get_yscale() == "log":
        # A perplexity ratio of 1.2 should read as "1.2", not "1.2 x 10^0" --
        # but only while the axis stays narrow. A collapsed run pushes the same
        # axis across nine decades, and there plain decimals are unreadable
        # ("0.00, 0.01, 1.00") where powers of ten are exactly right. Decide
        # from the data, not from the quantity's name.
        low, high = axis.get_ylim()
        if low > 0 and high / low < 100:
            axis.yaxis.set_major_formatter(ScalarFormatter())
            axis.yaxis.set_minor_formatter(ScalarFormatter())
    # The legend can land on the data -- which line it covers depends on the
    # measurements -- so it carries the surface behind it rather than being
    # placed at a position that happens to be empty for one run's numbers.
    axis.legend(
        fontsize=7,
        labelcolor=INK_SECONDARY,
        frameon=True,
        facecolor=SURFACE,
        edgecolor="none",
        framealpha=0.85,
    )


def write_plot(results: Mapping[str, Any], path: str) -> Optional[str]:
    """Four panels against unlearning strength; returns the path, or None.

    The panels are the four things the decision rule reads, in the order it
    reads them, so the figure and the verdict cannot drift apart. One y-axis per
    panel: perplexity ratios and AUCs never share a scale.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # One method per figure: two methods share no x axis, since "strength" is a
    # different hyperparameter for each and a line across them would assert a
    # trend that does not exist.
    by_method: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for setting in results["settings"]:
        if setting["strength"] > 0:
            by_method[setting["method"]].append(setting)
    if not by_method:
        return None
    method, settings = max(by_method.items(), key=lambda kv: len(kv[1]))
    settings = sorted(settings, key=lambda s: s["strength"])
    if len(settings) < 2:
        return None

    x = [s["strength"] for s in settings]
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), facecolor=SURFACE)
    for axis in axes.flat:
        axis.set_facecolor(SURFACE)

    # (a) membership -- the evidence that has to disappear
    axis = axes[0][0]
    for index, task in enumerate(("forget_vs_unseen", "member_vs_unseen")):
        _series(
            axis,
            x,
            [s["auc"].get(task, {}).get(MEMBERSHIP_SCORE) for s in settings],
            f"{MEMBERSHIP_SCORE} · {task}",
            index,
        )
    axis.axhspan(*MEMBERSHIP_GONE_BAND, color=THRESHOLD_COLOR, alpha=0.10, zorder=0)
    axis.axhline(0.5, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=1)
    _style(axis, "a. Membership evidence (must vanish into the band)", "AUC")

    # (b) deletion -- the signal under test
    axis = axes[0][1]
    plotted = [DELETION_SCORE] + [
        b
        for b in ("alignment_cos_sq", "loss_ratio_vs_finetuned", "grad_ratio")
        if any(b in s["auc"].get(DELETION_TASK, {}) for s in settings)
    ]
    for index, score in enumerate(plotted[: len(SERIES_COLORS)]):
        _series(
            axis,
            x,
            [s["auc"].get(DELETION_TASK, {}).get(score) for s in settings],
            score,
            index,
        )
    axis.axhline(
        DELETION_MIN_AUC, color=THRESHOLD_COLOR, linewidth=1.2, linestyle="--", zorder=1
    )
    axis.axhline(0.5, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=1)
    _style(axis, "b. Deletion signal on forget_vs_retain (threshold 0.70)", "AUC")

    # (c) utility -- what must survive
    axis = axes[1][0]
    for index, population in enumerate(UTILITY_POPULATIONS):
        _series(
            axis,
            x,
            [s["perplexity_ratio"][population] for s in settings],
            f"{population} perplexity ratio",
            index,
            log_y=True,
        )
    axis.axhline(
        UTILITY_MAX_PPL_RATIO,
        color=THRESHOLD_COLOR,
        linewidth=1.2,
        linestyle="--",
        zorder=1,
    )
    _style(axis, "c. Utility: unlearned / fine-tuned perplexity (max 1.10)", "ratio")

    # (d) forgetting -- how far the intervention actually got.
    # One measure only: the forget perplexity belongs to a different scale, and
    # putting it here would be a second measure sharing one axis. It is in the
    # report's tables instead.
    axis = axes[1][1]
    _series(
        axis,
        x,
        [s["forget_retain_loss_ratio"] for s in settings],
        "forget / retain loss ratio",
        0,
    )
    axis.axhline(
        FORGET_RATIO_TARGET,
        color=THRESHOLD_COLOR,
        linewidth=1.2,
        linestyle="--",
        zorder=1,
    )
    axis.axhline(1.0, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=1)
    _style(
        axis,
        f"d. Forgetting achieved (GA+GD stop target {FORGET_RATIO_TARGET:g}, dashed)",
        "forget loss / retain loss",
    )

    n_seeds = sorted({s["n_seeds"] for s in settings})
    figure.suptitle(
        f"{method} unlearning-strength sweep — mean over "
        f"{'/'.join(str(n) for n in n_seeds)} seeds, error bars = across-seed SD",
        fontsize=12,
        color=INK_PRIMARY,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return path


# ------------------------------------------------------------------------- CLI


def jsonable(value: Any) -> Any:
    """Non-finite floats become null: here they always mean "not computed"."""
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


def expand_inputs(patterns: Sequence[str]) -> List[str]:
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
        description="Summarise an unlearning-strength sweep against its "
                    "preregistered decision rule.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  $PY -m deletion_signals.sweep_report \\\n"
            "      --inputs 'ds_sweep/ascent_*/seed*/ga_gd/per_sample_scores.json' \\\n"
            "      --output ds_sweep/analysis\n"
        ),
    )
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Score files or glob patterns")
    parser.add_argument("--output", required=True,
                        help="Output directory for sweep_results.json, "
                             "sweep_report.md and sweep.png")
    parser.add_argument("--orientations", default=None,
                        help="Override the orientation table path")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip the figure (text output only)")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    table = OrientationTable.load(args.orientations)

    files = [load_any(path) for path in expand_inputs(args.inputs)]
    results = build_results(files, table, args.n_bootstrap, args.confidence)

    os.makedirs(args.output, exist_ok=True)
    results_path = os.path.join(args.output, "sweep_results.json")
    with open(results_path, "w") as handle:
        json.dump(jsonable(results), handle, indent=2, allow_nan=False)

    report = format_report(results)
    report_path = os.path.join(args.output, "sweep_report.md")
    with open(report_path, "w") as handle:
        handle.write(report)

    plot_path = None
    if not args.no_plot:
        plot_path = write_plot(results, os.path.join(args.output, "sweep.png"))

    if not args.quiet:
        print(report)
    print(f"\nWrote {results_path}")
    print(f"Wrote {report_path}")
    if plot_path:
        print(f"Wrote {plot_path}")
    elif not args.no_plot:
        print("No figure written: a sweep needs at least two strengths to plot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
