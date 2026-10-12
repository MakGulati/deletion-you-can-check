"""Low-prevalence security metrics: what a detector is worth in operation.

Why ROC-AUC is not the number that matters
-------------------------------------------
Every table in this study so far reports AUC on ``forget_vs_retain``, where 200
forget records sit against 600 retain records -- a deletion prevalence of 1 in 4.
No real audit looks like that. A model owner asked "were these records deleted?"
is searching a corpus in which the deletion set is 1 in 100 or rarer, and
precision at that prevalence is not a monotone relabelling of AUC: it collapses
with the base rate while AUC, which is invariant to it, does not move at all.

A detector with AUC 0.94 can be almost useless at 1:100 if its false-positive
rate is a few percent, because the false positives outnumber the true ones. That
gap between a headline AUC and an operational precision is what this module
measures.

Everything here is computed OFFLINE from saved per-sample score files. No model
is loaded and nothing is re-run: every detector this study defines is already
present in the score tables, so the entire component costs no GPU time.

How prevalence adjustment is done
---------------------------------
Analytically, from the measured operating point::

    precision(π) = π·TPR / ( π·TPR + (1-π)·FPR )
    recall(π)    = TPR

where TPR and FPR are measured on the observed populations at a fixed threshold
and π is the simulated prevalence. **Recall is prevalence-invariant; precision is
not.** That asymmetry is the finding, not an artefact of the formula.

The analytic route is preferred over subsampling negatives to hit a target
prevalence, for three reasons: it is deterministic and exactly reproducible, it
does not introduce an RNG whose seed would have to be recorded, and it does not
discard negatives that were actually measured in order to estimate the same
quantity with more variance. A resampling cross-check is computed beside it as a
diagnostic, and :func:`resampled_precision` exists so the agreement can be
verified rather than asserted.

The resolution floor, and why zero is not reported as perfection
-----------------------------------------------------------------
With ``n`` negatives the finest non-zero FPR the empirical ROC can express is
``1/n``. Two consequences are enforced here rather than left to the reader:

* An FPR target below ``1/n`` is **unresolvable** and is reported as ``None``
  with the reason attached. At the 600 retain samples this study scores, 1% FPR
  is resolvable (6 negatives) and 0.1% is not. That is declared in
  ``preregistration_v3_fllm.yaml`` in advance, so its absence is a preregistered
  outcome rather than a gap.
* An *observed* FPR of exactly zero would make ``precision(π) = 1.0`` at every
  prevalence -- a claim of perfect precision resting on having seen no false
  positive in 600 draws. The FPR is floored at ``1/n`` and the row is marked
  ``resolution_limited``.

Orientations are taken from ``configs/orientations.yaml`` and applied as
declared. Nothing here computes ``max(auc, 1-auc)`` or flips a score after
seeing how it performed.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .orientations import OrientationTable, build_task_arrays
from .reproducibility import new_rng, run_provenance
from .schema import SCORE_COLUMNS, ScoreFile, load_any
from .stats import average_precision, finitize, roc_auc, summarise_across_seeds, tpr_at_fpr

# positives : negatives in the audited pool.
DEFAULT_PREVALENCES: Tuple[str, ...] = ("1:1", "1:10", "1:100")
DEFAULT_FPR_TARGETS: Tuple[float, ...] = (0.01, 0.001)
DEFAULT_TASK = "forget_vs_retain"


# ------------------------------------------------------------------ prevalence


def parse_prevalence(label: str) -> float:
    """``"1:100"`` -> the positive base rate π = 1/101.

    Expressed as a ratio rather than a probability because that is how a
    deletion audit is described ("one deleted record in a hundred"), and
    converting it here keeps the arithmetic in one place instead of in every
    caller.
    """
    text = str(label).strip()
    if ":" not in text:
        value = float(text)
        if not 0.0 < value < 1.0:
            raise ValueError(
                f"A bare prevalence must be a probability in (0, 1), got {label!r}"
            )
        return value

    left, right = text.split(":", 1)
    positives, negatives = float(left), float(right)
    if positives <= 0 or negatives <= 0:
        raise ValueError(f"Prevalence {label!r} must have positive terms on both sides")
    return positives / (positives + negatives)


def prevalence_adjusted_precision(
    tpr: float, fpr: float, prevalence: float
) -> Optional[float]:
    """precision = π·TPR / (π·TPR + (1-π)·FPR).

    Returns ``None`` when the detector flags nothing at this operating point
    (TPR and FPR both zero): precision is undefined on an empty selection, and
    reporting 0.0 would say "everything it flagged was wrong" about a detector
    that flagged nothing.
    """
    if not (math.isfinite(tpr) and math.isfinite(fpr)):
        return None
    if not 0.0 < prevalence < 1.0:
        raise ValueError(f"prevalence must lie in (0, 1), got {prevalence}")

    true_positives = prevalence * tpr
    false_positives = (1.0 - prevalence) * fpr
    denominator = true_positives + false_positives
    if denominator <= 0.0:
        return None
    return true_positives / denominator


def resampled_precision(
    values: Sequence[float],
    labels: Sequence[int],
    threshold: float,
    prevalence: float,
    n_resamples: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Subsampling cross-check on :func:`prevalence_adjusted_precision`.

    Draws pools at the target prevalence, applies the same threshold, and
    reports the mean and SD of the realised precision. A **diagnostic only** --
    the analytic value is what the tables carry. It exists so the analytic
    formula can be verified against a finite-sample simulation rather than
    trusted, and so the extra variance that subsampling would have introduced is
    visible.
    """
    rng = rng or np.random.default_rng(0)
    arr = finitize(values)
    labels_arr = np.asarray(labels)
    pos = arr[labels_arr == 1]
    neg = arr[labels_arr == 0]
    if pos.size == 0 or neg.size == 0:
        return {"mean": None, "sd": None, "n_resamples": 0}

    # Hold the positive count and vary the negative count, so the simulated pool
    # keeps every positive that was actually measured.
    n_pos = pos.size
    n_neg = int(round(n_pos * (1.0 - prevalence) / prevalence))
    if n_neg <= 0:
        return {"mean": None, "sd": None, "n_resamples": 0}

    precisions: List[float] = []
    for _ in range(n_resamples):
        drawn_neg = rng.choice(neg, n_neg, replace=True)
        tp = int((pos >= threshold).sum())
        fp = int((drawn_neg >= threshold).sum())
        if tp + fp == 0:
            continue
        precisions.append(tp / (tp + fp))

    if not precisions:
        return {"mean": None, "sd": None, "n_resamples": 0}
    array = np.asarray(precisions, dtype=float)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "n_resamples": int(array.size),
        "simulated_negatives": n_neg,
        "note": "diagnostic cross-check; the analytic value is authoritative",
    }


# -------------------------------------------------------------- operating points


@dataclass
class OperatingPoint:
    """One threshold, and everything measurable at it."""

    name: str
    threshold: Optional[float]
    tpr: Optional[float]
    fpr: Optional[float]
    fpr_used: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    n_selected: int
    n_positive: int
    n_negative: int
    resolution_limited: bool = False
    resolvable: bool = True
    note: str = ""
    prevalence: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "threshold": _jsonable_float(self.threshold),
            "tpr": _jsonable_float(self.tpr),
            "fpr_observed": _jsonable_float(self.fpr),
            "fpr_used": _jsonable_float(self.fpr_used),
            "precision": _jsonable_float(self.precision),
            "recall": _jsonable_float(self.recall),
            "n_selected": self.n_selected,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "resolution_limited": self.resolution_limited,
            "resolvable": self.resolvable,
            "note": self.note,
            "prevalence": self.prevalence,
        }


def fpr_resolution(n_negative: int) -> float:
    """The finest non-zero FPR ``n_negative`` samples can express."""
    if n_negative <= 0:
        return float("nan")
    return 1.0 / float(n_negative)


def operating_point_at_top_k(
    values: Sequence[float], labels: Sequence[int], k: Optional[int] = None
) -> OperatingPoint:
    """Take the k highest-scoring samples, where k defaults to the positive count.

    This is the operationally natural rule when the deletion-set size is known:
    an auditor who knows 200 records were requested for deletion selects the 200
    most suspicious. Precision and recall coincide numerically at that k -- both
    are ``hits / k`` -- and both are reported, because "of what I flagged, how
    much was right" and "of what was deleted, how much did I find" are different
    questions that happen to share an answer here.

    Ties are broken by rank order rather than by score, so a detector that
    assigns many samples the same value does not silently select more than k.
    """
    arr = finitize(values)
    labels_arr = np.asarray(labels)
    n_positive = int((labels_arr == 1).sum())
    n_negative = int((labels_arr == 0).sum())
    k = int(k if k is not None else n_positive)

    if arr.size == 0 or n_positive == 0 or n_negative == 0 or k <= 0:
        return OperatingPoint(
            name=f"top_{k}", threshold=None, tpr=None, fpr=None, fpr_used=None,
            precision=None, recall=None, n_selected=0, n_positive=n_positive,
            n_negative=n_negative, resolvable=False, note="degenerate labels",
        )

    k = min(k, arr.size)
    order = np.argsort(-arr, kind="stable")[:k]
    selected = labels_arr[order]
    hits = int((selected == 1).sum())
    false_hits = int((selected == 0).sum())

    tpr = hits / n_positive
    observed_fpr = false_hits / n_negative
    floor = fpr_resolution(n_negative)
    resolution_limited = observed_fpr == 0.0
    fpr_used = floor if resolution_limited else observed_fpr

    return OperatingPoint(
        name=f"top_{k}",
        threshold=float(arr[order[-1]]),
        tpr=tpr,
        fpr=observed_fpr,
        fpr_used=fpr_used,
        precision=hits / k,
        recall=tpr,
        n_selected=k,
        n_positive=n_positive,
        n_negative=n_negative,
        resolution_limited=resolution_limited,
        note=(
            f"observed FPR is exactly 0 over {n_negative} negatives; floored at "
            f"the resolution {floor:.5f} so prevalence-adjusted precision is not "
            f"reported as 1.0"
            if resolution_limited
            else ""
        ),
    )


def operating_point_at_fpr(
    values: Sequence[float], labels: Sequence[int], target_fpr: float
) -> OperatingPoint:
    """The highest-TPR threshold whose FPR does not exceed ``target_fpr``.

    Unresolvable targets -- those below ``1/n_negative`` -- return an
    unpopulated point marked ``resolvable=False`` rather than a number. At that
    scale the statistic is the TPR at zero false positives, set by whichever
    single negative happened to score highest, and reporting it would invite it
    to be read as a measurement.
    """
    from sklearn.metrics import roc_curve

    arr = finitize(values)
    labels_arr = np.asarray(labels)
    n_positive = int((labels_arr == 1).sum())
    n_negative = int((labels_arr == 0).sum())
    name = f"fpr_{target_fpr:g}"

    estimate = tpr_at_fpr(arr, labels_arr, target_fpr)
    if not estimate.resolvable:
        return OperatingPoint(
            name=name, threshold=None, tpr=None, fpr=None, fpr_used=None,
            precision=None, recall=None, n_selected=0, n_positive=n_positive,
            n_negative=n_negative, resolvable=False, note=estimate.note,
        )

    fpr_curve, tpr_curve, thresholds = roc_curve(labels_arr, arr)
    eligible = np.flatnonzero(fpr_curve <= target_fpr)
    index = int(eligible[np.argmax(tpr_curve[eligible])])
    threshold = float(thresholds[index])
    tpr = float(tpr_curve[index])
    observed_fpr = float(fpr_curve[index])

    floor = fpr_resolution(n_negative)
    resolution_limited = observed_fpr == 0.0
    fpr_used = floor if resolution_limited else observed_fpr
    n_selected = int((arr >= threshold).sum())

    return OperatingPoint(
        name=name,
        threshold=threshold,
        tpr=tpr,
        fpr=observed_fpr,
        fpr_used=fpr_used,
        precision=(
            (tpr * n_positive) / n_selected if n_selected else None
        ),
        recall=tpr,
        n_selected=n_selected,
        n_positive=n_positive,
        n_negative=n_negative,
        resolution_limited=resolution_limited,
        note=(
            f"observed FPR is exactly 0 over {n_negative} negatives; floored at "
            f"the resolution {floor:.5f}"
            if resolution_limited
            else ""
        ),
    )


def attach_prevalence(
    point: OperatingPoint,
    prevalences: Sequence[str],
    values: Optional[Sequence[float]] = None,
    labels: Optional[Sequence[int]] = None,
    rng: Optional[np.random.Generator] = None,
    resample: bool = False,
) -> OperatingPoint:
    """Fill in prevalence-adjusted precision and recall at one operating point."""
    if point.tpr is None or point.fpr_used is None:
        point.prevalence = {
            label: {"precision": None, "recall": None, "reason": point.note or "no operating point"}
            for label in prevalences
        }
        return point

    block: Dict[str, Any] = {}
    for label in prevalences:
        pi = parse_prevalence(label)
        entry: Dict[str, Any] = {
            "base_rate": pi,
            "precision": prevalence_adjusted_precision(point.tpr, point.fpr_used, pi),
            # Stated explicitly on every row, because the single most common
            # misreading of this table is to expect recall to fall with
            # prevalence too.
            "recall": point.tpr,
            "recall_is_prevalence_invariant": True,
        }
        if point.resolution_limited:
            entry["resolution_limited"] = True
        if resample and values is not None and labels is not None and point.threshold is not None:
            entry["resampled"] = resampled_precision(
                values, labels, point.threshold, pi, rng=rng
            )
        block[label] = entry

    point.prevalence = block
    return point


# ------------------------------------------------------------- per-file scoring


def _jsonable_float(value: Optional[float]) -> Optional[float]:
    """Strict-JSON float: a non-finite value becomes null, meaning 'not computed'."""
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return as_float if math.isfinite(as_float) else None


def evaluate_detector(
    rows: Sequence[Mapping[str, Any]],
    detector: str,
    table: OrientationTable,
    task_name: str = DEFAULT_TASK,
    prevalences: Sequence[str] = DEFAULT_PREVALENCES,
    fpr_targets: Sequence[float] = DEFAULT_FPR_TARGETS,
    n_bootstrap: int = 0,
    seed: int = 0,
    resample: bool = False,
) -> Optional[Dict[str, Any]]:
    """Every low-prevalence metric for one detector on one scored checkpoint.

    Returns ``None`` when the detector is absent from these rows -- which is a
    real case: the confirmatory GA+GD runs were scored with FUMA and Fisher-SIF
    switched off, and an absent column must stay absent rather than be filled
    with a value that would enter the rankings.
    """
    task = table.task(task_name)
    values, labels, dropped = build_task_arrays(rows, detector, task)
    if not values or len(set(labels)) < 2:
        return None

    orientation = table.orientation(detector, task_name)
    oriented = table.orient(values, detector, task_name)
    labels_arr = np.asarray(labels)
    n_positive = int((labels_arr == 1).sum())
    n_negative = int((labels_arr == 0).sum())

    entry: Dict[str, Any] = {
        "detector": detector,
        "task": task_name,
        "direction": orientation.direction,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_dropped": dropped,
        "fpr_resolution": fpr_resolution(n_negative),
        "roc_auc": _jsonable_float(roc_auc(oriented, labels)),
        "average_precision": _jsonable_float(average_precision(oriented, labels)),
        "tpr_at_fpr": {},
        "operating_points": {},
    }

    for target in fpr_targets:
        estimate = tpr_at_fpr(oriented, labels, target)
        entry["tpr_at_fpr"][f"{target:g}"] = estimate.as_dict()

    rng = new_rng(seed)
    points = [operating_point_at_top_k(oriented, labels, n_positive)]
    for target in fpr_targets:
        points.append(operating_point_at_fpr(oriented, labels, target))

    for point in points:
        attach_prevalence(
            point, prevalences, oriented, labels, rng=rng, resample=resample
        )
        entry["operating_points"][point.name] = point.as_dict()

    # The top-k point is the headline: it is the rule an auditor who knows the
    # deletion-set size would actually apply.
    top_k = entry["operating_points"][f"top_{n_positive}"]
    entry["precision_at_deletion_set_size"] = top_k["precision"]
    entry["recall_at_deletion_set_size"] = top_k["recall"]
    entry["top_k_recall"] = top_k["recall"]
    entry["k"] = n_positive

    if n_bootstrap:
        from .stats import bootstrap_auc_ci

        entry["bootstrap"] = bootstrap_auc_ci(
            oriented, labels, n_bootstrap=n_bootstrap, rng=new_rng(seed)
        ).as_dict()
        entry["bootstrap"]["caveat"] = (
            "within-run precision over samples; NOT a substitute for the "
            "across-seed replication that the seed-level summary provides"
        )

    return entry


def evaluate_file(
    score_file: ScoreFile,
    table: OrientationTable,
    detectors: Sequence[str],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Every requested detector on one score file."""
    from .audit import setting_label

    metadata = score_file.metadata
    method = str(metadata.get("method", "unknown"))
    seed = int(metadata.get("seed", -1))

    results: Dict[str, Any] = {}
    missing: List[str] = []
    for detector in detectors:
        entry = evaluate_detector(
            score_file.rows, detector, table, seed=seed, **kwargs
        )
        if entry is None:
            missing.append(detector)
        else:
            results[detector] = entry

    return {
        "path": score_file.path,
        "method": method,
        "setting": setting_label(metadata, method),
        "seed": seed,
        "populations": metadata.get("populations", {}),
        "detectors": results,
        "detectors_not_collected": missing,
        "not_collected_note": (
            "these score columns are absent from this run's configuration and "
            "are reported as absent, never imputed"
            if missing
            else ""
        ),
    }


# ----------------------------------------------------------------- aggregation


def _metric_paths() -> Dict[str, Sequence[str]]:
    """The scalar metrics that get summarised across seeds, and where they live."""
    return {
        "roc_auc": ("roc_auc",),
        "average_precision": ("average_precision",),
        "precision_at_deletion_set_size": ("precision_at_deletion_set_size",),
        "recall_at_deletion_set_size": ("recall_at_deletion_set_size",),
        "top_k_recall": ("top_k_recall",),
        "tpr_at_fpr_0.01": ("tpr_at_fpr", "0.01", "tpr"),
        "tpr_at_fpr_0.001": ("tpr_at_fpr", "0.001", "tpr"),
    }


def _dig(entry: Mapping[str, Any], path: Sequence[str]) -> Optional[float]:
    cursor: Any = entry
    for key in path:
        if not isinstance(cursor, Mapping) or key not in cursor:
            return None
        cursor = cursor[key]
    return _jsonable_float(cursor) if isinstance(cursor, (int, float)) else None


def aggregate(evaluations: Sequence[Mapping[str, Any]],
              prevalences: Sequence[str] = DEFAULT_PREVALENCES) -> Dict[str, Any]:
    """Group per-file evaluations by (method, setting) and summarise over seeds.

    The seed is the replication unit throughout: per-seed values are kept and
    reported, and the mean and SD are computed over seeds, never over samples.
    A setting evaluated on one seed reports an SD of NaN rather than 0.
    """
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for evaluation in evaluations:
        grouped.setdefault(
            (evaluation["method"], evaluation["setting"]), []
        ).append(evaluation)

    settings: List[Dict[str, Any]] = []
    for (method, setting), files in sorted(grouped.items()):
        detectors = sorted({d for f in files for d in f["detectors"]})
        block: Dict[str, Any] = {}
        for detector in detectors:
            per_seed_entries = {
                int(f["seed"]): f["detectors"][detector]
                for f in files
                if detector in f["detectors"]
            }
            summary: Dict[str, Any] = {
                "n_seeds": len(per_seed_entries),
                "seeds": sorted(per_seed_entries),
            }
            for name, path in _metric_paths().items():
                values = {
                    seed: _dig(entry, path)
                    for seed, entry in per_seed_entries.items()
                }
                usable = {s: v for s, v in values.items() if v is not None}
                summary[name] = (
                    summarise_across_seeds(usable)
                    if usable
                    else {"mean": None, "sd": None, "n_seeds": 0, "per_seed": {}}
                )

            prevalence_block: Dict[str, Any] = {}
            for label in prevalences:
                precisions = {}
                for seed, entry in per_seed_entries.items():
                    top_k = next(
                        (v for k, v in entry["operating_points"].items()
                         if k.startswith("top_")),
                        None,
                    )
                    if top_k is None:
                        continue
                    value = (top_k.get("prevalence", {}).get(label) or {}).get("precision")
                    if value is not None:
                        precisions[seed] = float(value)
                prevalence_block[label] = {
                    "base_rate": parse_prevalence(label),
                    "precision_at_top_k": (
                        summarise_across_seeds(precisions)
                        if precisions
                        else {"mean": None, "sd": None, "n_seeds": 0, "per_seed": {}}
                    ),
                }
            summary["prevalence"] = prevalence_block

            # Any detector absent from ANY seed is flagged, so a mean is never
            # read as if it covered all three when it covered two.
            summary["complete_across_seeds"] = len(per_seed_entries) == len(files)
            block[detector] = summary

        settings.append(
            {
                "method": method,
                "setting": setting,
                "n_files": len(files),
                "seeds": sorted(int(f["seed"]) for f in files),
                "detectors": block,
                "detectors_not_collected": sorted(
                    {d for f in files for d in f["detectors_not_collected"]}
                ),
            }
        )

    return {"settings": settings}


# -------------------------------------------------------------------- the CLI


def build_results(
    paths: Sequence[str],
    orientations_path: Optional[str] = None,
    detectors: Optional[Sequence[str]] = None,
    task: str = DEFAULT_TASK,
    prevalences: Sequence[str] = DEFAULT_PREVALENCES,
    fpr_targets: Sequence[float] = DEFAULT_FPR_TARGETS,
    n_bootstrap: int = 0,
    resample: bool = False,
) -> Dict[str, Any]:
    """Evaluate every score file and assemble the component's result document."""
    table = OrientationTable.load(orientations_path)
    detectors = list(detectors or SCORE_COLUMNS)

    evaluations: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for path in paths:
        try:
            score_file = load_any(path)
        except (OSError, ValueError) as exc:
            failures.append({"path": path, "error": str(exc)})
            continue
        try:
            evaluations.append(
                evaluate_file(
                    score_file,
                    table,
                    detectors,
                    task_name=task,
                    prevalences=prevalences,
                    fpr_targets=fpr_targets,
                    n_bootstrap=n_bootstrap,
                    resample=resample,
                )
            )
        except (KeyError, ValueError) as exc:
            failures.append({"path": path, "error": str(exc)})

    return {
        "component": "B -- low-prevalence security metrics",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "prevalences": {label: parse_prevalence(label) for label in prevalences},
        "fpr_targets": list(fpr_targets),
        "statistical_unit": "seed",
        "n_inputs": len(paths),
        "orientations": table.source_path,
        "method_notes": {
            "prevalence_adjustment": (
                "analytic: precision(pi) = pi*TPR / (pi*TPR + (1-pi)*FPR), with "
                "TPR and FPR measured at a fixed threshold on the observed "
                "populations. Recall is prevalence-invariant."
            ),
            "fpr_floor": (
                "an observed FPR of exactly 0 is replaced by 1/n_negative, the "
                "finest value the sample supports, and the row is marked "
                "resolution_limited"
            ),
            "unresolvable_targets": (
                "an FPR target below 1/n_negative is reported as null with its "
                "reason, never as a number"
            ),
            "orientation": (
                "taken from the preregistered table; no score is flipped after "
                "its performance is observed"
            ),
        },
        "per_file": evaluations,
        "aggregate": aggregate(evaluations, prevalences),
        "failures": failures,
        "provenance": run_provenance(),
    }


CSV_COLUMNS = (
    "method", "setting", "detector", "n_seeds", "complete_across_seeds",
    "roc_auc_mean", "roc_auc_sd",
    "average_precision_mean", "average_precision_sd",
    "tpr_at_1pct_fpr_mean", "tpr_at_1pct_fpr_sd",
    "precision_at_k_mean", "precision_at_k_sd",
    "recall_at_k_mean", "recall_at_k_sd",
    "precision_1to1", "precision_1to10", "precision_1to100",
)


def write_csv(results: Mapping[str, Any], path: str) -> str:
    """Flat per-(setting, detector) table, mean and SD over seeds."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def mean_of(block: Mapping[str, Any], key: str, field_name: str = "mean") -> str:
        value = (block.get(key) or {}).get(field_name)
        return "" if value is None or not math.isfinite(float(value)) else f"{float(value):.6f}"

    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for setting in results["aggregate"]["settings"]:
            for detector, block in sorted(setting["detectors"].items()):
                prevalence = block.get("prevalence", {})

                def prevalence_mean(label: str) -> str:
                    entry = (prevalence.get(label) or {}).get("precision_at_top_k") or {}
                    value = entry.get("mean")
                    return "" if value is None else f"{float(value):.6f}"

                writer.writerow(
                    [
                        setting["method"],
                        setting["setting"],
                        detector,
                        block.get("n_seeds", 0),
                        block.get("complete_across_seeds", False),
                        mean_of(block, "roc_auc"),
                        mean_of(block, "roc_auc", "sd"),
                        mean_of(block, "average_precision"),
                        mean_of(block, "average_precision", "sd"),
                        mean_of(block, "tpr_at_fpr_0.01"),
                        mean_of(block, "tpr_at_fpr_0.01", "sd"),
                        mean_of(block, "precision_at_deletion_set_size"),
                        mean_of(block, "precision_at_deletion_set_size", "sd"),
                        mean_of(block, "recall_at_deletion_set_size"),
                        mean_of(block, "recall_at_deletion_set_size", "sd"),
                        prevalence_mean("1:1"),
                        prevalence_mean("1:10"),
                        prevalence_mean("1:100"),
                    ]
                )
    return path


def expand_inputs(patterns: Sequence[str]) -> List[str]:
    """Resolve globs, de-duplicated and sorted so a run is reproducible."""
    paths: List[str] = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern, recursive=True))
        if not matched and os.path.exists(pattern):
            matched = [pattern]
        paths.extend(matched)
    return sorted(dict.fromkeys(os.path.abspath(p) for p in paths))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Low-prevalence security metrics for deletion detectors, computed "
            "offline from saved per-sample score files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  $PY -m deletion_signals.lowprev \\\n"
            "      --inputs 'ds_confirm/**/per_sample_scores.json' \\\n"
            "               'fllm_audit/**/per_sample_scores.json' \\\n"
            "      --output fllm_audit/lowprev\n"
        ),
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", default="./fllm_audit/lowprev")
    parser.add_argument("--orientations", default=None)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--detectors", nargs="+", default=None,
                        help=f"Default: every score column ({len(SCORE_COLUMNS)})")
    parser.add_argument("--prevalences", nargs="+", default=list(DEFAULT_PREVALENCES))
    parser.add_argument("--fpr-targets", type=float, nargs="+",
                        default=list(DEFAULT_FPR_TARGETS))
    parser.add_argument("--n-bootstrap", type=int, default=0,
                        help="Within-run sample bootstrap; 0 disables it")
    parser.add_argument("--resample-check", action="store_true",
                        help="Also compute the subsampling cross-check on the "
                             "analytic prevalence adjustment (diagnostic)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    paths = expand_inputs(args.inputs)
    if not paths:
        raise SystemExit(f"No score files matched {args.inputs}")

    results = build_results(
        paths,
        orientations_path=args.orientations,
        detectors=args.detectors,
        task=args.task,
        prevalences=args.prevalences,
        fpr_targets=args.fpr_targets,
        n_bootstrap=args.n_bootstrap,
        resample=args.resample_check,
    )

    os.makedirs(args.output, exist_ok=True)
    json_path = os.path.join(args.output, "lowprev_results.json")
    with open(json_path, "w") as handle:
        # allow_nan=False: this document is strict JSON, where null means "not
        # computed" and there is no other way to say it.
        json.dump(results, handle, indent=2, allow_nan=False, default=str)
    csv_path = write_csv(results, os.path.join(args.output, "lowprev_summary.csv"))

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    if results["failures"]:
        print(f"\n{len(results['failures'])} input(s) failed; see 'failures' in the JSON")
    return 0


if __name__ == "__main__":
    sys.exit(main())
