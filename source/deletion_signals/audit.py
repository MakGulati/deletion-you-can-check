"""The three-axis unlearning audit.

One question, three axes that only mean something together:

===================  ===================  =========================================
axis                 comparison           question
===================  ===================  =========================================
membership removal   forget vs test       do forgotten records still look like
                                          training members?
deletion specificity forget vs retain     can an auditor identify which members
                                          were targeted?
utility              retain/test, before  is the model still useful?
                     vs after
===================  ===================  =========================================

Why jointly
-----------
Because membership evidence disappears for two completely different reasons, and
one axis cannot tell them apart. In the exploratory sweep, three settings of five
reached the chance region on membership -- and all three had destroyed the model,
raising retain perplexity by seven orders of magnitude. A membership-only
evaluation would have certified every one of them as successful unlearning.

Selective forgetting, and why not the loss ratio
------------------------------------------------
The forget/retain loss ratio reads 1.049 for the best viable GA+GD setting and
1.005 for a globally destroyed one: near-identical numbers for opposite
outcomes, because both losses sit near 2.0 in one case and both are enormous in
the other. It also hides the real behaviour -- that setting moves the forget set
37x more than the retain set, so it is *specific*; what it lacks is magnitude.

:func:`rnff` measures magnitude against the only meaningful yardstick, the exact
retraining control::

    RNFF(P) = [ L_post(P) - L_finetuned(P) ] / [ L_retrain(P) - L_finetuned(P) ]

RNFF_forget = 1 means "moved the forget set exactly as far as deleting it
would have". RNFF_retain is the specificity check, and global destruction fails
it by construction, because destruction moves retain as far as it moves forget.

Thresholds
----------
Read from ``configs/preregistration_v2.yaml``, never hardcoded here, so a number
in this module cannot drift from the document that declares it.
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
from .sweep_report import perplexity, ratio, token_weighted_mean_nll

DEFAULT_PREREGISTRATION = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configs", "preregistration_v2.yaml"
)

EXACT_RETRAIN = "exact_retrain"
NULL_REFIT = "null_refit"
# The FLLM audit's nuisance conditions: a checkpoint that changed for a
# legitimate reason other than deletion. They are controls in the same sense the
# other two are -- their detector scores are floors, not capabilities -- so they
# are labelled and grouped as controls rather than as methods under test. Only
# EXACT_RETRAIN ever supplies the RNFF denominator; adding names here does not
# change that.
ROUTINE_UPDATE = "routine_update"
EXACT_RETRAIN_ROUTINE_UPDATE = "exact_retrain_routine_update"
CONTROL_METHODS = (
    EXACT_RETRAIN,
    NULL_REFIT,
    ROUTINE_UPDATE,
    EXACT_RETRAIN_ROUTINE_UPDATE,
)


# --------------------------------------------------------------- preregistration


@dataclass
class Preregistration:
    """The declared thresholds, loaded from YAML and never edited in code."""

    raw: Dict[str, Any]
    path: str

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Preregistration":
        import yaml

        path = path or DEFAULT_PREREGISTRATION
        with open(path, "r") as handle:
            raw = yaml.safe_load(handle)
        if int(raw.get("schema_version", 0)) != 2:
            raise ValueError(
                f"{path} declares schema_version={raw.get('schema_version')!r}; "
                f"this module reads version 2."
            )
        for axis in ("membership_removal", "utility_preservation", "selective_forgetting"):
            if axis not in raw.get("axes", {}):
                raise ValueError(f"{path} declares no '{axis}' axis")
        return cls(raw=raw, path=path)

    # -- axes -------------------------------------------------------------
    @property
    def membership(self) -> Dict[str, Any]:
        return self.raw["axes"]["membership_removal"]

    @property
    def utility(self) -> Dict[str, Any]:
        return self.raw["axes"]["utility_preservation"]

    @property
    def selectivity(self) -> Dict[str, Any]:
        return self.raw["axes"]["selective_forgetting"]

    @property
    def deletion(self) -> Dict[str, Any]:
        return self.raw["decision_rule"]["deletion_detectable"]

    @property
    def primary_detector(self) -> str:
        return self.raw["primary_detector"]

    @property
    def primary_task(self) -> str:
        return self.raw["primary_task"]

    @property
    def detectors(self) -> Dict[str, Any]:
        return self.raw["detectors"]

    def access_of(self, detector: str) -> List[str]:
        return list(self.detectors.get(detector, {}).get("access", []))

    def weaker_or_equal_access(self, detector: str, other: str) -> bool:
        """True when ``other``'s access is a subset of ``detector``'s.

        The matched-access rule: a complex detector may be claimed as
        advantageous only over detectors whose access it already subsumes.
        Beating a detector that needs *more* than you do is not evidence of
        anything.
        """
        return set(self.access_of(other)) <= set(self.access_of(detector))


# ------------------------------------------------------------------- the metric


def rnff(
    loss_post: Optional[float],
    loss_finetuned: Optional[float],
    denominator: Optional[float],
    min_denominator: float = 1e-3,
) -> Optional[float]:
    """How far a population moved, in units of ``denominator``.

    ``denominator`` is always the **forget set's** gold-standard change,
    ``L_retrain(forget) - L_finetuned(forget)``, for *both* populations. That is
    deliberate and it is the only well-posed choice.

    Normalising D_retain by its own retraining change does not work: exact
    retraining trains on D_retain, so that denominator is ~0 by construction and
    the ratio is undefined in precisely the ordinary case. A specificity clause
    built on it would never fire -- including for a model that had been
    destroyed, which is the one case it exists to catch.

    With the shared denominator both numbers read on one scale, "fraction of the
    intended deletion effect":

    * ``RNFF_forget`` -- how much of the intended effect was achieved.
    * ``RNFF_retain`` -- how much collateral movement it cost, in the same
      units. Global destruction moves retain as far as forget, so this rises
      with RNFF_forget and the specificity clause fails, which is exactly what
      the forget/retain loss ratio could not see.
    """
    if loss_post is None or loss_finetuned is None or denominator is None:
        return None
    if not all(math.isfinite(v) for v in (loss_post, loss_finetuned, denominator)):
        return None
    if abs(denominator) < min_denominator:
        # Exact retraining barely moved the forget set. The scale the whole
        # criterion is expressed in does not exist for this seed, and inventing
        # one would manufacture precision.
        return None
    return (loss_post - loss_finetuned) / denominator


def rnff_denominator(
    retrain_losses: Optional[Mapping[str, float]],
    finetuned_forget_loss: Optional[float],
) -> Optional[float]:
    """``L_retrain(forget) - L_finetuned(forget)`` -- the gold-standard change.

    The distance a forget sample's loss *should* travel if the record were
    genuinely deleted, measured on this seed's own retraining control.
    """
    if not retrain_losses or finetuned_forget_loss is None:
        return None
    retrain_forget = retrain_losses.get("forget")
    if retrain_forget is None or not math.isfinite(retrain_forget):
        return None
    if not math.isfinite(finetuned_forget_loss):
        return None
    return retrain_forget - finetuned_forget_loss


def population_losses(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Optional[float]]]:
    """Token-weighted mean NLL per population, under both resident models."""
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for population in POPULATIONS:
        subset = [r for r in rows if str(r.get("population")) == population]
        n_tokens = [r.get("n_tokens") for r in subset]
        out[population] = {
            key: token_weighted_mean_nll([r.get(column) for r in subset], n_tokens)[0]
            for key, column in (("post", "loss_unlearned"), ("finetuned", "loss_finetuned"))
        }
    return out


def retrain_losses_by_seed(files: Sequence[ScoreFile]) -> Dict[int, Dict[str, float]]:
    """L_retrain per population per seed, from the exact-retraining control.

    Keyed by seed and never pooled: seed 42's methods are normalised by seed
    42's own retraining control, because the two share a split and a fine-tuned
    starting point. Normalising by another seed's control would divide by a
    denominator measured on different data.
    """
    out: Dict[int, Dict[str, float]] = {}
    for score_file in files:
        method = str(score_file.metadata.get("method", ""))
        if method != EXACT_RETRAIN:
            continue
        seed = int(score_file.metadata.get("seed", -1))
        losses = population_losses(score_file.rows)
        out[seed] = {
            population: losses[population]["post"]
            for population in POPULATIONS
            if losses[population]["post"] is not None
        }
    return out


# ------------------------------------------------------------------------ runs


@dataclass
class AuditedRun:
    """One scored checkpoint, on all three axes."""

    path: str
    seed: int
    method: str
    setting: str
    is_control: bool = False
    counts: Dict[str, int] = field(default_factory=dict)
    losses: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)
    perplexity_ratio: Dict[str, Optional[float]] = field(default_factory=dict)
    rnff: Dict[str, Optional[float]] = field(default_factory=dict)
    rnff_denominator: Optional[float] = None
    aucs: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    unlearning: Dict[str, Any] = field(default_factory=dict)
    data_quality: Dict[str, Any] = field(default_factory=dict)
    axes: Dict[str, Any] = field(default_factory=dict)

    def auc(self, task: str, score: str) -> Optional[float]:
        entry = self.aucs.get(task, {}).get(score)
        return None if entry is None else entry.get("auc")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "seed": self.seed,
            "method": self.method,
            "setting": self.setting,
            "is_control": self.is_control,
            "counts": self.counts,
            "mean_token_nll": self.losses,
            "perplexity_ratio": self.perplexity_ratio,
            "rnff": self.rnff,
            "rnff_denominator": self.rnff_denominator,
            "auc": self.aucs,
            "unlearning": self.unlearning,
            "data_quality": self.data_quality,
            "axes": self.axes,
        }


def setting_label(metadata: Mapping[str, Any], method: str) -> str:
    """A label that distinguishes runs which must not be pooled.

    Grouping is what decides which runs get averaged together, so this label has
    to carry *every* dimension a family varies -- not just the obvious swept
    scalar. A stabilisation study varies which constraint is active while
    holding the learning rate fixed; keying on the learning rate alone would
    average five different interventions into one row and silently destroy the
    comparison the block exists to make.
    """
    if method in CONTROL_METHODS:
        return method

    run = metadata.get("unlearning_run") or {}
    hyperparameters = run.get("hyperparameters") or {}

    # Stabilised GA+GD: the arm is defined by which constraint is switched on.
    stabilizers = run.get("stabilizers")
    if stabilizers:
        active = []
        if float(stabilizers.get("retain_kl_weight") or 0) > 0:
            active.append(f"kl={float(stabilizers['retain_kl_weight']):g}")
        if stabilizers.get("trust_region_radius"):
            active.append(f"trust_region={float(stabilizers['trust_region_radius']):g}")
        if float(stabilizers.get("max_grad_norm") or 1.0) != 1.0:
            active.append(f"clip={float(stabilizers['max_grad_norm']):g}")
        if stabilizers.get("utility_guard_ratio"):
            active.append(f"guard={float(stabilizers['utility_guard_ratio']):g}")
        arm = "+".join(active) if active else "unconstrained"
        return f"{arm} @ ascent_lr={float(hyperparameters.get('ascent_lr', 0)):g}"

    # NPO sweeps the learning rate at fixed beta, but beta belongs in the label
    # anyway: a later block that varies it must not pool with this one, and a
    # label carrying only the swept scalar is exactly how that happens.
    if method == "npo" or "retain_weight" in hyperparameters:
        return (
            f"lr={float(hyperparameters.get('learning_rate', 0)):g}, "
            f"beta={float(hyperparameters.get('beta', 0)):g}"
        )

    # SCRUB varies the forget weight and the step size independently.
    if "forget_kl_weight" in hyperparameters:
        return (
            f"forget_kl={float(hyperparameters['forget_kl_weight']):g}, "
            f"lr={float(hyperparameters['learning_rate']):g}"
        )

    for key in ("ascent_lr", "learning_rate"):
        if hyperparameters.get(key) is not None:
            return f"{key}={float(hyperparameters[key]):g}"
    return method


def data_quality(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Counts a reader needs before trusting any AUC in the same table."""
    fingerprints = {
        population: {r["text_sha1"] for r in rows if r["population"] == population}
        for population in POPULATIONS
    }
    overlaps = {
        f"{a}&{b}": len(fingerprints[a] & fingerprints[b])
        for i, a in enumerate(POPULATIONS)
        for b in POPULATIONS[i + 1 :]
    }
    nulls: Dict[str, int] = {}
    nonfinite: Dict[str, int] = {}
    for column in SCORE_COLUMNS:
        null_count = 0
        nonfinite_count = 0
        for row in rows:
            value = row.get(column)
            if value is None:
                null_count += 1
                continue
            try:
                as_float = float(value)
            except (TypeError, ValueError):
                nonfinite_count += 1
                continue
            # -inf is a legitimate ranking position for combined_log_score only.
            if math.isnan(as_float) or (math.isinf(as_float) and column != "combined_log_score"):
                nonfinite_count += 1
        if null_count:
            nulls[column] = null_count
        if nonfinite_count:
            nonfinite[column] = nonfinite_count

    return {
        "n_rows": len(rows),
        "counts": {p: len(fingerprints[p]) for p in POPULATIONS},
        "population_overlaps": overlaps,
        "populations_disjoint": all(v == 0 for v in overlaps.values()),
        "null_counts": nulls,
        "nonfinite_counts": nonfinite,
        "truncated": sum(1 for r in rows if (r.get("flags") or {}).get("truncated")),
        "nonfinite_loss_flagged": sum(
            1 for r in rows if (r.get("flags") or {}).get("nonfinite_loss")
        ),
        "zero_grad_norm_flagged": sum(
            1 for r in rows if (r.get("flags") or {}).get("zero_grad_norm")
        ),
    }


def audit_run(
    score_file: ScoreFile,
    table: OrientationTable,
    prereg: Preregistration,
    retrain_losses: Mapping[int, Mapping[str, float]],
    n_bootstrap: int,
    confidence: float,
) -> AuditedRun:
    """Every quantity the three-axis rule needs, for one scored checkpoint."""
    rows = score_file.rows
    if not rows:
        raise ValueError(f"{score_file.path} contains no rows")

    metadata = score_file.metadata
    method = str(metadata.get("method") or rows[0].get("method", "unknown"))
    seed = int(metadata.get("seed", rows[0].get("seed", -1)))

    run = AuditedRun(
        path=score_file.path or "",
        seed=seed,
        method=method,
        setting=setting_label(metadata, method),
        is_control=method in CONTROL_METHODS,
    )
    run.data_quality = data_quality(rows)
    run.counts = run.data_quality["counts"]
    run.losses = population_losses(rows)

    # One denominator for every population: the forget set's gold-standard
    # change under this seed's own retraining control.
    denominator = rnff_denominator(
        retrain_losses.get(seed), run.losses["forget"]["finetuned"]
    )
    run.rnff_denominator = denominator
    for population in POPULATIONS:
        entry = run.losses[population]
        run.perplexity_ratio[population] = ratio(
            perplexity(entry["post"]), perplexity(entry["finetuned"])
        )
        run.rnff[population] = rnff(entry["post"], entry["finetuned"], denominator)

    unlearning_run = metadata.get("unlearning_run") or {}
    run.unlearning = {
        key: unlearning_run.get(key)
        for key in (
            "steps_configured", "steps_taken", "stop_reason",
            "final_forget_loss", "final_retain_loss", "trajectory_path",
            "control", "training_seed", "n_training_samples",
        )
    }

    available = set(score_file.available_scores)
    tasks = [prereg.primary_task, prereg.membership["task"]] + list(
        prereg.membership.get("supporting_tasks", [])
    )
    needs_ci = {
        (prereg.primary_task, prereg.primary_detector),
        (prereg.membership["task"], prereg.membership["score"]),
    }

    for task_name in dict.fromkeys(tasks):
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
                "access": prereg.access_of(score),
                "n_dropped": dropped,
            }
            if (task_name, score) in needs_ci:
                entry.update(
                    bootstrap_auc_ci(
                        oriented, labels, n_bootstrap=n_bootstrap,
                        confidence=confidence, rng=new_rng(seed),
                    ).as_dict()
                )
            else:
                entry["auc"] = roc_auc(oriented, labels)
            block[score] = entry
        run.aucs[task_name] = block

    run.axes = evaluate_axes(run, prereg)
    return run


def evaluate_axes(run: AuditedRun, prereg: Preregistration) -> Dict[str, Any]:
    """Apply the three preregistered axes to one run.

    The deletion axis is deliberately NOT decided here: its third clause
    compares against the null-refit control, which is a property of the study
    rather than of a single run. :func:`classify` completes it.
    """
    membership = prereg.membership
    low, high = membership["equivalence_region"]
    membership_auc = run.auc(membership["task"], membership["score"])
    membership_removed = (
        membership_auc is not None
        and math.isfinite(membership_auc)
        and low <= membership_auc <= high
    )

    utility = prereg.utility
    limit = float(utility["max_perplexity_ratio"])
    utility_ratios = {p: run.perplexity_ratio.get(p) for p in utility["populations"]}
    known = [v for v in utility_ratios.values() if v is not None and math.isfinite(v)]
    utility_preserved = (
        len(known) == len(utility["populations"]) and all(v <= limit for v in known)
    )

    selectivity = prereg.selectivity
    forget_rnff = run.rnff.get("forget")
    retain_rnff = run.rnff.get("retain")
    # Both clauses must be MEASURED. An absent RNFF_retain means the shared
    # denominator was unavailable, so specificity is unknown -- and unknown is
    # not the same as satisfied. Treating it as satisfied is how a destroyed
    # model slips through, since destruction and success both leave the
    # criterion unevaluated if the denominator is missing.
    selective = (
        forget_rnff is not None
        and math.isfinite(forget_rnff)
        and forget_rnff >= float(selectivity["min_forget"])
        and retain_rnff is not None
        and math.isfinite(retain_rnff)
        and abs(retain_rnff) <= float(selectivity["max_retain"])
    )

    deletion = prereg.deletion
    detector = prereg.primary_detector
    entry = run.aucs.get(prereg.primary_task, {}).get(detector, {})

    return {
        "membership_removed": {
            "met": bool(membership_removed),
            "score": membership["score"],
            "task": membership["task"],
            "auc": membership_auc,
            "equivalence_region": [low, high],
        },
        "utility_preserved": {
            "met": bool(utility_preserved),
            "perplexity_ratio": utility_ratios,
            "max_allowed": limit,
            "evaluable": len(known) == len(utility["populations"]),
        },
        "selective_forgetting": {
            "met": bool(selective),
            "rnff_forget": forget_rnff,
            "rnff_retain": retain_rnff,
            "min_forget": selectivity["min_forget"],
            "max_retain": selectivity["max_retain"],
            "denominator": run.rnff_denominator,
            "evaluable": forget_rnff is not None and retain_rnff is not None,
        },
        "deletion_detectable": {
            "detector": detector,
            "task": prereg.primary_task,
            "auc": entry.get("auc"),
            "ci_lower": entry.get("ci_lower"),
            "min_auc": deletion["min_auc"],
            "ci_lower_above": deletion["bootstrap_ci_lower_above"],
            # completed by classify(), which knows the null-refit floor
            "met": None,
        },
    }


# -------------------------------------------------------------------- the study


def control_validity(runs: Sequence[AuditedRun]) -> Dict[str, Any]:
    """Is the RNFF denominator measuring deletion, or training variance?

    RNFF divides by ``L_retrain(forget) - L_finetuned(forget)``, which is taken
    to be the loss change caused by *removing the forget set*. That reading is
    only sound if two independently trained models would not have produced a
    comparable change on their own -- and the null refit measures exactly that,
    having deleted nothing.

    So the diagnostic is ``RNFF_forget`` evaluated on the null refit:

    * near 0  -- the denominator is deletion, and RNFF means what it says;
    * far from 0 -- the denominator is contaminated by run-to-run variation,
      every RNFF in the study is inflated by roughly that amount, and the
      criterion needs a difference-in-differences correction before it can be
      read. That is a change to the preregistration, not to the analysis, and
      it must be versioned rather than applied silently.

    Reported, never used to alter a verdict: this is information a reader needs
    in order to know how much to trust the selectivity axis.
    """
    null_runs = [r for r in runs if r.method == NULL_REFIT]
    if not null_runs:
        return {
            "evaluable": False,
            "reason": "no null_refit control loaded; the RNFF denominator's "
                      "attribution to deletion is unverified",
        }

    forget = summarise_across_seeds(
        {r.seed: r.rnff["forget"] for r in null_runs if r.rnff.get("forget") is not None}
    )
    retain = summarise_across_seeds(
        {r.seed: r.rnff["retain"] for r in null_runs if r.rnff.get("retain") is not None}
    )
    mean = forget.get("mean")
    contaminated = mean is not None and math.isfinite(mean) and abs(mean) > 0.20

    return {
        "evaluable": True,
        "null_refit_rnff_forget": forget,
        "null_refit_rnff_retain": retain,
        "denominator_contaminated": bool(contaminated),
        "interpretation": (
            "the null refit deleted nothing, so its RNFF_forget of "
            f"{mean:.3f} is run-to-run training variation appearing as "
            "'forgetting'. Every RNFF in this study is inflated by roughly that "
            "amount and the selectivity axis should not be read at face value "
            "until the denominator is corrected for it."
            if contaminated else
            "the null refit, which deleted nothing, shows RNFF_forget near "
            "zero, so the denominator is attributable to deletion rather than "
            "to training variance."
        ) if mean is not None and math.isfinite(mean) else
        "the null refit produced no measurable RNFF; the denominator's "
        "attribution is unverified.",
    }


def detector_floors(runs: Sequence[AuditedRun], task: str) -> Dict[str, Dict[str, Any]]:
    """Each detector's false-positive floor, from the null-refit control.

    Nothing was deleted in those runs, so whatever a detector scores there is
    produced by run-to-run variation between two independently trained models.
    A detector cannot be said to have found deletion until it clears this.
    """
    null_runs = [r for r in runs if r.method == NULL_REFIT]
    detectors = sorted({s for r in null_runs for s in r.aucs.get(task, {})})
    floors: Dict[str, Dict[str, Any]] = {}
    for detector in detectors:
        summary = summarise_across_seeds(
            {r.seed: r.auc(task, detector) for r in null_runs if r.auc(task, detector) is not None}
        )
        floors[detector] = summary
    return floors


def classify(
    run: AuditedRun,
    prereg: Preregistration,
    floors: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Complete the deletion axis, then apply the mechanical classification."""
    axes = run.axes
    deletion = axes["deletion_detectable"]
    rule = prereg.deletion

    auc = deletion.get("auc")
    ci_lower = deletion.get("ci_lower")
    clears_thresholds = (
        auc is not None
        and math.isfinite(auc)
        and auc >= float(rule["min_auc"])
        and ci_lower is not None
        and math.isfinite(ci_lower)
        and ci_lower > float(rule["bootstrap_ci_lower_above"])
    )

    floor_value = None
    clears_floor = True
    if rule.get("must_exceed_null_refit"):
        floor = floors.get(prereg.primary_detector) or {}
        mean, sd = floor.get("mean"), floor.get("sd")
        if mean is not None and math.isfinite(mean):
            margin = float(rule.get("null_refit_margin_sds", 2.0))
            sd_value = sd if (sd is not None and math.isfinite(sd)) else 0.0
            floor_value = mean + margin * sd_value
            clears_floor = auc is not None and math.isfinite(auc) and auc > floor_value
        else:
            # No null control loaded: the floor is unknown, so the clause cannot
            # be evaluated. Refusing it is the conservative reading -- an
            # unverified detector is not a verified one.
            clears_floor = False

    deletion["null_refit_floor"] = floor_value
    deletion["clears_thresholds"] = bool(clears_thresholds)
    deletion["clears_null_floor"] = bool(clears_floor)
    deletion["met"] = bool(clears_thresholds and clears_floor)

    membership = axes["membership_removed"]["met"]
    utility = axes["utility_preserved"]["met"]
    selective = axes["selective_forgetting"]["met"]
    detectable = deletion["met"]

    if membership and utility and selective and detectable:
        label = "persistent_fingerprint"
    elif membership and utility and selective and not detectable:
        label = "successful_deletion_no_fingerprint"
    elif membership and not utility:
        label = "global_destruction"
    elif utility and not membership:
        label = "incomplete_forgetting"
    else:
        label = "inconclusive"

    return {
        "classification": label,
        "statement": prereg.raw["classifications"][label]["statement"].strip(),
        "axes_met": {
            "membership_removed": membership,
            "utility_preserved": utility,
            "selective_forgetting": selective,
            "deletion_detectable": detectable,
        },
    }


def summarise_setting(runs: Sequence[AuditedRun], prereg: Preregistration) -> Dict[str, Any]:
    """Across-seed summary of one (method, setting)."""
    first = runs[0]
    summary: Dict[str, Any] = {
        "method": first.method,
        "setting": first.setting,
        "is_control": first.is_control,
        "seeds": sorted(r.seed for r in runs),
        "n_seeds": len(runs),
        "per_seed": {
            r.seed: {
                "classification": r.axes.get("_verdict", {}).get("classification"),
                "membership_auc": r.axes["membership_removed"]["auc"],
                "perplexity_ratio": r.perplexity_ratio,
                "rnff": r.rnff,
                "deletion_auc": r.axes["deletion_detectable"]["auc"],
                "stop_reason": r.unlearning.get("stop_reason"),
            }
            for r in runs
        },
    }

    summary["perplexity_ratio"] = {
        p: summarise_across_seeds(
            {r.seed: r.perplexity_ratio[p] for r in runs if r.perplexity_ratio.get(p) is not None}
        )
        for p in POPULATIONS
    }
    summary["rnff"] = {
        p: summarise_across_seeds(
            {r.seed: r.rnff[p] for r in runs if r.rnff.get(p) is not None}
        )
        for p in POPULATIONS
    }

    tasks = sorted({t for r in runs for t in r.aucs})
    summary["auc"] = {
        task: {
            score: summarise_across_seeds(
                {r.seed: r.auc(task, score) for r in runs if r.auc(task, score) is not None}
            )
            for score in sorted({s for r in runs for s in r.aucs.get(task, {})})
        }
        for task in tasks
    }

    # Matched-access paired comparison: the primary detector against every
    # detector whose access it subsumes. Only these differences can support a
    # claim that the more complex detector is worth its access.
    primary = prereg.primary_detector
    task = prereg.primary_task
    comparisons = []
    for other in sorted({s for r in runs for s in r.aucs.get(task, {})}):
        if other == primary:
            continue
        differences = {
            r.seed: r.auc(task, primary) - r.auc(task, other)
            for r in runs
            if r.auc(task, primary) is not None and r.auc(task, other) is not None
        }
        if not differences:
            continue
        comparisons.append({
            "primary": primary,
            "other": other,
            "other_access": prereg.access_of(other),
            "other_access_is_weaker_or_equal": prereg.weaker_or_equal_access(primary, other),
            **summarise_across_seeds(differences),
        })
    summary["paired_differences"] = comparisons

    verdicts = [r.axes.get("_verdict", {}).get("classification") for r in runs]
    summary["classifications"] = {v: verdicts.count(v) for v in sorted(set(v for v in verdicts if v))}
    summary["unanimous_classification"] = (
        verdicts[0] if verdicts and len(set(verdicts)) == 1 else None
    )
    summary["axes_met_seed_counts"] = {
        axis: sum(1 for r in runs if r.axes.get("_verdict", {}).get("axes_met", {}).get(axis))
        for axis in ("membership_removed", "utility_preserved",
                     "selective_forgetting", "deletion_detectable")
    }
    summary["candidate_operating_point"] = bool(
        summary["axes_met_seed_counts"]["membership_removed"] == len(runs)
        and summary["axes_met_seed_counts"]["utility_preserved"] == len(runs)
        and summary["axes_met_seed_counts"]["selective_forgetting"] == len(runs)
    )
    return summary


def build_audit(
    files: Sequence[ScoreFile],
    table: OrientationTable,
    prereg: Preregistration,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """The whole audit: every run on three axes, grouped and classified."""
    retrain = retrain_losses_by_seed(files)

    runs = [
        audit_run(f, table, prereg, retrain, n_bootstrap, confidence) for f in files
    ]
    floors = detector_floors(runs, prereg.primary_task)

    for run in runs:
        run.axes["_verdict"] = classify(run, prereg, floors)

    grouped: Dict[Tuple[str, str], List[AuditedRun]] = defaultdict(list)
    for run in runs:
        grouped[(run.method, run.setting)].append(run)

    settings = [
        summarise_setting(sorted(group, key=lambda r: r.seed), prereg)
        for _, group in sorted(grouped.items())
    ]

    candidates = [
        s for s in settings
        if s["candidate_operating_point"] and not s["is_control"]
    ]

    return {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "preregistration": prereg.path,
            "preregistration_schema": prereg.raw.get("schema_version"),
            "inputs": [f.path for f in files],
            "n_bootstrap": n_bootstrap,
            "confidence": confidence,
            "replication_unit": "seed",
            "primary_detector": prereg.primary_detector,
            "primary_task": prereg.primary_task,
            "retrain_control_seeds": sorted(retrain),
            "null_refit_floors": floors,
            "control_validity": control_validity(runs),
            "thresholds": {
                "membership_equivalence_region": prereg.membership["equivalence_region"],
                "max_perplexity_ratio": prereg.utility["max_perplexity_ratio"],
                "rnff_min_forget": prereg.selectivity["min_forget"],
                "rnff_max_retain": prereg.selectivity["max_retain"],
                "deletion_min_auc": prereg.deletion["min_auc"],
            },
            "provenance": run_provenance(),
        },
        "runs": [r.as_dict() for r in sorted(runs, key=lambda r: (r.method, r.setting, r.seed))],
        "settings": settings,
        "verdict": {
            "candidate_operating_points": [
                {"method": s["method"], "setting": s["setting"]} for s in candidates
            ],
            "outcome": _outcome(settings, candidates, prereg),
        },
    }


def _outcome(
    settings: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    prereg: Preregistration,
) -> str:
    """The study-level statement, chosen mechanically from the classifications."""
    if candidates:
        with_fingerprint = [
            s for s in candidates
            if s["axes_met_seed_counts"]["deletion_detectable"] == s["n_seeds"]
        ]
        if with_fingerprint:
            return prereg.raw["classifications"]["persistent_fingerprint"]["statement"].strip()
        return prereg.raw["classifications"]["successful_deletion_no_fingerprint"]["statement"].strip()

    methods = [s for s in settings if not s["is_control"]]
    labels = {s["unanimous_classification"] for s in methods}

    if labels == {"incomplete_forgetting"}:
        return prereg.raw["classifications"]["incomplete_forgetting"]["statement"].strip()
    if labels == {"global_destruction"}:
        return prereg.raw["classifications"]["global_destruction"]["statement"].strip()
    if {"global_destruction", "incomplete_forgetting"} <= labels:
        # The pattern the exploratory sweep suggested, and the one a
        # membership-only evaluation is least equipped to see: each setting
        # fails on a different axis, so every axis looks satisfiable, and none
        # is satisfiable at the same time as the others.
        return (
            "remember-or-collapse: every setting either left forgotten records "
            "looking like training members, or removed the membership evidence "
            "only by destroying the model. No setting did both. "
            + prereg.raw["classifications"]["global_destruction"]["statement"].strip()
        )
    return prereg.raw["classifications"]["inconclusive"]["statement"].strip()


# ------------------------------------------------------------------------- CLI


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


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Three-axis unlearning audit against the confirmatory "
                    "preregistration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  $PY -m deletion_signals.audit \\\n"
            "      --inputs 'ds_confirm/**/per_sample_scores.json' \\\n"
            "      --output ds_confirm/audit\n"
        ),
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preregistration", default=None)
    parser.add_argument("--orientations", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .audit_report import format_report

    args = parse_args(argv)
    prereg = Preregistration.load(args.preregistration)
    table = OrientationTable.load(args.orientations)

    files = [load_any(path) for path in expand_inputs(args.inputs)]
    results = build_audit(files, table, prereg, args.n_bootstrap, args.confidence)

    os.makedirs(args.output, exist_ok=True)
    results_path = os.path.join(args.output, "audit_results.json")
    with open(results_path, "w") as handle:
        json.dump(jsonable(results), handle, indent=2, allow_nan=False)

    report = format_report(results, prereg)
    report_path = os.path.join(args.output, "audit_report.md")
    with open(report_path, "w") as handle:
        handle.write(report)

    if not args.quiet:
        print(report)
    print(f"\nWrote {results_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
