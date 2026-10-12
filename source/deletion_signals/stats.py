"""Metrics, confidence intervals and comparisons.

Two levels of uncertainty appear in this evaluation and they answer different
questions:

* **Within a run** -- bootstrap over samples. Answers "given this trained and
  unlearned model, how precisely is the AUC estimated". Resampling samples does
  NOT license a claim about the method, because every sample shares one trained
  model.
* **Across runs** -- paired comparison over seeds. Answers "does score A beat
  score B for this unlearning method". The experimental unit is the seed (an
  independently fine-tuned and unlearned model), never a sample within one
  model. :func:`across_seed_comparison` is the only function here whose output
  supports a claim of that kind, and it refuses to pretend three seeds are
  many.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


def finitize(scores: Sequence[float]) -> np.ndarray:
    """Map -inf onto a finite value below every real score, preserving rank.

    sklearn rejects non-finite input, but -inf is meaningful for
    ``combined_log_score``: it marks a sample with zero alignment, which sorts
    below every sample that has some. Replacing it with (min finite - 1) keeps
    the ranking exactly, and an AUC depends on nothing else. NaN is left in
    place so it fails loudly -- a NaN score is a bug, not a ranking position.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return arr
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    return np.where(np.isneginf(arr), finite.min() - 1.0, arr)


def roc_auc(values: Sequence[float], labels: Sequence[int]) -> float:
    """ROC AUC with ``values`` already oriented so that larger means positive."""
    from sklearn.metrics import roc_auc_score

    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(np.asarray(labels), finitize(values)))


def average_precision(values: Sequence[float], labels: Sequence[int]) -> float:
    """Average precision (area under the precision-recall curve)."""
    from sklearn.metrics import average_precision_score

    if len(set(labels)) < 2:
        return float("nan")
    return float(average_precision_score(np.asarray(labels), finitize(values)))


@dataclass
class TprAtFpr:
    """TPR at a target FPR, with an explicit note when the target is unreachable."""

    target_fpr: float
    tpr: float
    n_negative: int
    resolvable: bool
    note: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "target_fpr": self.target_fpr,
            "tpr": None if not self.resolvable else self.tpr,
            "n_negative": self.n_negative,
            "resolvable": self.resolvable,
            "note": self.note,
        }


def tpr_at_fpr(
    values: Sequence[float], labels: Sequence[int], target_fpr: float
) -> TprAtFpr:
    """Highest TPR achievable at FPR <= ``target_fpr``.

    With ``n`` negatives the finest non-zero FPR the empirical ROC can express
    is ``1/n``. Asking for 0.1% FPR against a few hundred negatives therefore
    collapses to "TPR at exactly zero false positives", a single-sample
    statistic whose value is dominated by the one highest-scoring negative. That
    case is marked ``resolvable=False`` and reported as ``None`` rather than as
    a number that would be read as a measurement.
    """
    from sklearn.metrics import roc_curve

    labels_arr = np.asarray(labels)
    n_negative = int((labels_arr == 0).sum())

    if len(set(labels)) < 2 or n_negative == 0:
        return TprAtFpr(target_fpr, float("nan"), n_negative, False, "degenerate labels")

    if n_negative * target_fpr < 1.0:
        return TprAtFpr(
            target_fpr,
            float("nan"),
            n_negative,
            False,
            f"unresolvable: {n_negative} negatives give a minimum non-zero FPR of "
            f"{1.0 / n_negative:.4f} > {target_fpr}; any value here would be the "
            f"TPR at zero false positives, set by a single negative sample",
        )

    fpr, tpr, _ = roc_curve(labels_arr, finitize(values))
    eligible = tpr[fpr <= target_fpr]
    return TprAtFpr(
        target_fpr, float(eligible.max()) if eligible.size else 0.0, n_negative, True
    )


@dataclass
class AucEstimate:
    """A point AUC with a percentile bootstrap interval over samples."""

    auc: float
    ci_lower: float
    ci_upper: float
    n_positive: int
    n_negative: int
    n_bootstrap: int
    confidence: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "auc": self.auc,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "n_bootstrap": self.n_bootstrap,
            "confidence": self.confidence,
            "ci_kind": "percentile bootstrap over samples within one run",
        }


def bootstrap_auc_ci(
    values: Sequence[float],
    labels: Sequence[int],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> AucEstimate:
    """Observed AUC plus a stratified percentile bootstrap interval.

    The point estimate is the OBSERVED AUC, not the bootstrap mean: the two
    differ, and reporting the resampling mean under the name "AUC" misstates the
    estimate. Resampling is stratified within class so the class balance -- and
    hence the quantity being estimated -- is held fixed across replicates.
    """
    from sklearn.metrics import roc_auc_score

    rng = rng or np.random.default_rng(0)
    values_arr = finitize(values)
    labels_arr = np.asarray(labels)

    pos_idx = np.flatnonzero(labels_arr == 1)
    neg_idx = np.flatnonzero(labels_arr == 0)
    n_pos, n_neg = pos_idx.size, neg_idx.size

    if n_pos == 0 or n_neg == 0:
        return AucEstimate(
            float("nan"), float("nan"), float("nan"), n_pos, n_neg, 0, confidence
        )

    observed = float(roc_auc_score(labels_arr, values_arr))

    boot_labels = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
    replicates: List[float] = []
    for _ in range(n_bootstrap):
        sample = np.concatenate(
            [
                values_arr[rng.choice(pos_idx, n_pos, replace=True)],
                values_arr[rng.choice(neg_idx, n_neg, replace=True)],
            ]
        )
        try:
            replicates.append(float(roc_auc_score(boot_labels, sample)))
        except ValueError:
            continue

    if not replicates:
        return AucEstimate(
            observed, float("nan"), float("nan"), n_pos, n_neg, 0, confidence
        )

    alpha = 1.0 - confidence
    lower = float(np.percentile(replicates, 100 * alpha / 2))
    upper = float(np.percentile(replicates, 100 * (1 - alpha / 2)))
    return AucEstimate(observed, lower, upper, n_pos, n_neg, len(replicates), confidence)


def paired_bootstrap_auc_diff(
    values_a: Sequence[float],
    values_b: Sequence[float],
    labels: Sequence[int],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """Within-run paired bootstrap of AUC(A) - AUC(B) on the same samples.

    Both scores are resampled on the SAME bootstrap indices, so the shared
    sample-level noise cancels and the interval reflects only the difference
    between the two scoring rules on this one model.

    This is a within-run diagnostic. It cannot support "score A beats score B
    for this method" -- for that see :func:`across_seed_comparison`, whose
    experimental unit is the independently trained model.
    """
    from sklearn.metrics import roc_auc_score

    rng = rng or np.random.default_rng(0)
    a = finitize(values_a)
    b = finitize(values_b)
    labels_arr = np.asarray(labels)

    if a.size != b.size or a.size != labels_arr.size:
        raise ValueError(
            f"paired comparison needs aligned inputs; got {a.size}, {b.size}, "
            f"{labels_arr.size}"
        )

    pos_idx = np.flatnonzero(labels_arr == 1)
    neg_idx = np.flatnonzero(labels_arr == 0)
    if pos_idx.size == 0 or neg_idx.size == 0:
        return {"auc_a": float("nan"), "auc_b": float("nan"), "diff": float("nan")}

    auc_a = float(roc_auc_score(labels_arr, a))
    auc_b = float(roc_auc_score(labels_arr, b))

    boot_labels = np.concatenate(
        [np.ones(pos_idx.size, dtype=int), np.zeros(neg_idx.size, dtype=int)]
    )
    diffs: List[float] = []
    for _ in range(n_bootstrap):
        idx = np.concatenate(
            [
                rng.choice(pos_idx, pos_idx.size, replace=True),
                rng.choice(neg_idx, neg_idx.size, replace=True),
            ]
        )
        try:
            diffs.append(
                float(roc_auc_score(boot_labels, a[idx]))
                - float(roc_auc_score(boot_labels, b[idx]))
            )
        except ValueError:
            continue

    alpha = 1.0 - confidence
    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "diff": auc_a - auc_b,
        "ci_lower": float(np.percentile(diffs, 100 * alpha / 2)) if diffs else float("nan"),
        "ci_upper": float(np.percentile(diffs, 100 * (1 - alpha / 2))) if diffs else float("nan"),
        "n_bootstrap": len(diffs),
        "unit": "sample (within one trained model)",
        "caveat": (
            "within-run only: this interval does not account for variation "
            "between independently trained and unlearned models"
        ),
    }


@dataclass
class AcrossSeedComparison:
    """Paired comparison of two scores over matched seeds."""

    score_a: str
    score_b: str
    seeds: List[int] = field(default_factory=list)
    auc_a: List[float] = field(default_factory=list)
    auc_b: List[float] = field(default_factory=list)
    mean_diff: float = float("nan")
    sd_diff: float = float("nan")
    test_name: str = "none"
    statistic: float = float("nan")
    p_value: float = float("nan")
    note: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "score_a": self.score_a,
            "score_b": self.score_b,
            "seeds": self.seeds,
            "auc_a_per_seed": self.auc_a,
            "auc_b_per_seed": self.auc_b,
            "mean_diff": self.mean_diff,
            "sd_diff": self.sd_diff,
            "n_seeds": len(self.seeds),
            "test": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "unit": "seed (independently fine-tuned and unlearned model)",
            "note": self.note,
        }


def across_seed_comparison(
    score_a: str,
    score_b: str,
    auc_by_seed_a: Dict[int, float],
    auc_by_seed_b: Dict[int, float],
) -> AcrossSeedComparison:
    """Paired test of AUC(A) - AUC(B) across matched seeds.

    Only seeds present in both mappings are used, so an unfinished run for one
    score cannot unbalance the pairing.

    Test selection, and its honest limits at the scale this study runs at:

    * ``n < 2``  -- no test; the difference is reported without one.
    * ``n < 6``  -- a paired t-test on the per-seed differences. The
      distribution-free alternative (Wilcoxon signed-rank) cannot return
      p < 0.05 below six pairs: its smallest attainable two-sided p-value at
      n=3 is 0.25. Reporting "not significant" from that would say nothing
      about the data. The t-test's normality assumption over three points is
      itself unverifiable, so ``note`` says so and the mean difference with its
      across-seed SD is the number to read.
    * ``n >= 6`` -- Wilcoxon signed-rank, which needs no distributional
      assumption.
    """
    from scipy import stats as scipy_stats

    shared = sorted(set(auc_by_seed_a) & set(auc_by_seed_b))
    a = [auc_by_seed_a[s] for s in shared]
    b = [auc_by_seed_b[s] for s in shared]
    result = AcrossSeedComparison(score_a=score_a, score_b=score_b, seeds=shared, auc_a=a, auc_b=b)

    finite_pairs = [(x, y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    if not finite_pairs:
        result.note = "no seed has a finite AUC for both scores"
        return result

    diffs = np.array([x - y for x, y in finite_pairs], dtype=float)
    result.mean_diff = float(diffs.mean())
    result.sd_diff = float(diffs.std(ddof=1)) if diffs.size > 1 else float("nan")

    if diffs.size < 2:
        result.note = f"only {diffs.size} paired seed(s); no test performed"
        return result

    # A difference that is constant across seeds -- up to floating-point noise,
    # which a bare == 0.0 check on the SD would miss. A paired test divides by
    # the standard error and would return p ~ 0: certainty manufactured out of
    # having no between-seed variation to estimate from. Report the difference
    # and decline the test.
    sd = 0.0 if diffs.size < 2 else float(diffs.std(ddof=1))
    if sd <= 1e-9 * max(abs(result.mean_diff), 1e-12):
        result.sd_diff = sd
        result.test_name = "none"
        result.note = (
            "the two scores give identical AUCs on every seed"
            if abs(result.mean_diff) <= 1e-12
            else (
                f"the AUC difference is {result.mean_diff:.6g} on every seed, so "
                f"there is no between-seed variation to test against; a paired "
                f"test here would report p ~ 0 from zero degrees of variation"
            )
        )
        return result

    if diffs.size < 6:
        stat, p = scipy_stats.ttest_rel(
            [x for x, _ in finite_pairs], [y for _, y in finite_pairs]
        )
        result.test_name = "paired t-test"
        result.statistic = float(stat)
        result.p_value = float(p)
        result.note = (
            f"n={diffs.size} seeds. Wilcoxon signed-rank cannot reach p<0.05 below "
            f"6 pairs (min two-sided p at n=3 is 0.25), so a paired t-test is used "
            f"instead; its normality assumption is unverifiable at this n. Read the "
            f"mean difference and its across-seed SD as the primary evidence."
        )
        return result

    stat, p = scipy_stats.wilcoxon(
        [x for x, _ in finite_pairs], [y for _, y in finite_pairs]
    )
    result.test_name = "Wilcoxon signed-rank"
    result.statistic = float(stat)
    result.p_value = float(p)
    return result


def permutation_test_auc_above_chance(
    values: Sequence[float],
    labels: Sequence[int],
    n_permutations: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """Label-permutation test of H0: the oriented AUC is <= 0.5.

    The p-value is ``(k + 1) / (B + 1)``. The observed statistic is itself one
    draw from the null, so the estimate is bounded below by ``1/(B+1)`` instead
    of collapsing to a literal zero, which no finite Monte Carlo test can
    justify.
    """
    from sklearn.metrics import roc_auc_score

    rng = rng or np.random.default_rng(0)
    values_arr = finitize(values)
    labels_arr = np.asarray(labels)

    if len(set(labels)) < 2:
        return {"auc": float("nan"), "p_value": float("nan"), "n_permutations": 0}

    observed = float(roc_auc_score(labels_arr, values_arr))
    n_at_least = 0
    performed = 0
    for _ in range(n_permutations):
        permuted = rng.permutation(labels_arr)
        try:
            if float(roc_auc_score(permuted, values_arr)) >= observed:
                n_at_least += 1
            performed += 1
        except ValueError:
            continue

    return {
        "auc": observed,
        "p_value": (n_at_least + 1) / (performed + 1) if performed else float("nan"),
        "n_permutations": performed,
        "unit": "sample (within one trained model)",
    }


def spearman(x: Sequence[float], y: Sequence[float]) -> Dict[str, object]:
    """Spearman rank correlation over the pairs where both values are finite."""
    from scipy import stats as scipy_stats

    pairs = [
        (float(a), float(b))
        for a, b in zip(x, y)
        if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    if len(pairs) < 3:
        return {"rho": float("nan"), "p_value": float("nan"), "n": len(pairs)}

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return {"rho": float("nan"), "p_value": float("nan"), "n": len(pairs),
                "note": "one input is constant"}

    rho, p = scipy_stats.spearmanr(xs, ys)
    return {"rho": float(rho), "p_value": float(p), "n": len(pairs)}


def summarise_across_seeds(values_by_seed: Dict[int, float]) -> Dict[str, object]:
    """Mean, SD and n over seeds, ignoring seeds with a non-finite value."""
    finite = {s: v for s, v in values_by_seed.items() if v is not None and math.isfinite(v)}
    if not finite:
        return {"mean": float("nan"), "sd": float("nan"), "n_seeds": 0, "per_seed": {}}
    arr = np.array(list(finite.values()), dtype=float)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
        "n_seeds": int(arr.size),
        "per_seed": {int(s): float(v) for s, v in sorted(finite.items())},
    }
