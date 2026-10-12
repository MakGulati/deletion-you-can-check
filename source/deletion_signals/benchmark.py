"""What each detector needs, and what it costs to run.

Two questions this study cannot answer with AUC alone
------------------------------------------------------
**Access.** A detector that reads per-sample gradients through two checkpoints is
not competing with one that reads two loss values -- it is solving an easier
problem with more information. Comparing them as though they shared a threat
model is the single most common way a white-box result gets reported as though
it were an attack. :data:`ACCESS` states, for every detector, exactly what its
holder must possess, and :func:`comparable` refuses comparisons where the
challenger's access is not a superset of the baseline's.

**Cost.** Gradient detectors need a backward pass per sample; calibrated ones
need a second model resident. On the 200 forget and 600 retain samples this
study scores that is the difference between minutes and an hour, and between one
model in memory and four. A capability that costs 40x more for a 0.001 AUC gain
is not the same finding as one that costs nothing.

Measured primitives, composed costs
------------------------------------
Benchmarking each detector end to end would re-measure the same forward and
backward passes a dozen times over, which is both slow and misleading -- in the
real collector, ``grad_norm_sq``, ``raw_cosine``, ``alignment_cos_sq`` and
``sif`` all come out of ONE backward pass and none of them pays for it alone.

So the primitives are measured on the real models and the per-detector costs are
composed from them, with both reported and the composition shown:

* **measured** -- checkpoint load time per model, forward-pass time per sample,
  forward+backward time per sample, bf16 shadow backward time, the one-off
  direction construction, the one-off Fisher curvature build, and peak GPU
  memory for each.
* **composed** -- per detector, ``standalone`` (what it costs if it is the only
  thing you compute) and ``marginal`` (what it adds when the others are already
  being computed). Both matter and they differ by an order of magnitude for the
  gradient family, so reporting only one of them would misstate the comparison
  in whichever direction the reporter preferred.

Pass counts are asserted against the collector's actual call graph by
``tests/test_benchmark.py``, not taken from each detector's definition -- a
detector's formula says what it needs, not what the implementation does.

Everything in the top half of this module is pure data and pure arithmetic, so
it is unit-testable without a GPU. Only :func:`measure_primitives` needs one.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# ------------------------------------------------------------- access classes
#
# The fine-grained dimensions the FLLM preregistration enumerates. The coarse
# classes in preregistration_v2.yaml (unlearned / finetuned / base / gradients /
# curvature / lora_structure) map onto these; test_benchmark.py asserts the two
# tables do not contradict each other.

UNLEARNED = "unlearned_checkpoint"
BASE = "pretrained_base_checkpoint"
FINETUNED = "finetuned_pre_unlearning_checkpoint"
BOTH_CHECKPOINTS = "both_pre_and_post_checkpoints"
LOGITS = "logits_or_token_probabilities"
LABELS = "labels_or_target_tokens"
PER_EXAMPLE_LOSS = "per_example_loss"
PER_EXAMPLE_GRADIENTS = "per_example_gradients"
DISPLACEMENT = "global_model_displacement"
CURVATURE = "fisher_or_curvature"
AUXILIARY_DATA = "auxiliary_training_data"
ATTACK_CLASSIFIER = "trained_attack_classifier"

# Threat-model bands, used to label rows rather than to rank them.
AUDIT = "authorized_audit"
INSIDER = "insider_or_checkpoint_compromise"
BLACK_BOX = "ordinary_black_box_user"


@dataclass(frozen=True)
class DetectorSpec:
    """One detector's access requirements and its per-sample pass budget."""

    name: str
    access: Tuple[str, ...]
    # Passes through a model, per scored sample. A backward pass implies its
    # own forward pass; `backward` counts the backward half only, and
    # `forward` counts forward-only passes, so total forwards = forward + backward.
    forward_unlearned: int = 0
    backward_unlearned: int = 0
    forward_finetuned: int = 0
    backward_finetuned: int = 0
    forward_base: int = 0
    backward_shadow: int = 0
    needs_direction: bool = False
    needs_curvature: bool = False
    models_resident: Tuple[str, ...] = ()
    threat_band: str = AUDIT
    note: str = ""

    @property
    def forward_passes_per_sample(self) -> int:
        """Total forward passes, including the forward half of each backward."""
        return (
            self.forward_unlearned
            + self.forward_finetuned
            + self.forward_base
            + self.backward_passes_per_sample
        )

    @property
    def backward_passes_per_sample(self) -> int:
        return self.backward_unlearned + self.backward_finetuned + self.backward_shadow


# The table. Pass counts follow collect.py's documented call graph:
#   unlearned  1 bwd -> loss, ||grad||^2, cos, cos^2, SIF
#   base       1 fwd -> loss_base
#   finetuned  1 bwd -> ||grad_ref||^2
#   finetuned  1 fwd -> loss_finetuned
#   shadow     1 bwd -> both halves of the FUMA ratio
ACCESS: Dict[str, DetectorSpec] = {
    "loss_unlearned": DetectorSpec(
        name="loss_unlearned",
        access=(UNLEARNED, LOGITS, LABELS, PER_EXAMPLE_LOSS),
        forward_unlearned=1,
        models_resident=("unlearned",),
        threat_band=BLACK_BOX,
        note=(
            "the only detector here a party holding just the released model "
            "could compute, and then only with token-level output access"
        ),
    ),
    "loss_ratio": DetectorSpec(
        name="loss_ratio",
        access=(UNLEARNED, BASE, LOGITS, LABELS, PER_EXAMPLE_LOSS),
        forward_unlearned=1,
        forward_base=1,
        models_resident=("unlearned", "base"),
        note="difficulty-calibrated membership baseline; the base model is public",
    ),
    "loss_delta": DetectorSpec(
        name="loss_delta",
        access=(UNLEARNED, BASE, LOGITS, LABELS, PER_EXAMPLE_LOSS),
        forward_unlearned=1,
        forward_base=1,
        models_resident=("unlearned", "base"),
        note="additive form of loss_ratio; identical cost",
    ),
    "loss_ratio_vs_finetuned": DetectorSpec(
        name="loss_ratio_vs_finetuned",
        access=(UNLEARNED, FINETUNED, BOTH_CHECKPOINTS, LOGITS, LABELS, PER_EXAMPLE_LOSS),
        forward_unlearned=1,
        forward_finetuned=1,
        models_resident=("unlearned", "finetuned"),
        note=(
            "THE MATCHED-ACCESS BASELINE. Two forward passes, no gradients, no "
            "displacement vector. Every gradient detector below must beat this "
            "to support a gradient-specific claim."
        ),
    ),
    "loss_delta_vs_finetuned": DetectorSpec(
        name="loss_delta_vs_finetuned",
        access=(UNLEARNED, FINETUNED, BOTH_CHECKPOINTS, LOGITS, LABELS, PER_EXAMPLE_LOSS),
        forward_unlearned=1,
        forward_finetuned=1,
        models_resident=("unlearned", "finetuned"),
        note="additive form of the matched-access baseline; identical cost",
    ),
    # Every detector below computes a gradient, and a gradient cannot be
    # obtained without first computing the loss it is the gradient of. So
    # PER_EXAMPLE_LOSS is part of their access set whether or not they use the
    # value: access is what the holder POSSESSES, not what the formula consumes.
    # Omitting it would let a gradient detector fail the subset test against the
    # loss baseline it strictly dominates in information, and the matched-access
    # comparison -- the central question of Component C -- would never run.
    "grad_norm_sq": DetectorSpec(
        name="grad_norm_sq",
        access=(UNLEARNED, LOGITS, LABELS, PER_EXAMPLE_LOSS, PER_EXAMPLE_GRADIENTS),
        backward_unlearned=1,
        models_resident=("unlearned",),
        threat_band=INSIDER,
        note="raw gradient magnitude; needs white-box weights but only one checkpoint",
    ),
    "grad_ratio": DetectorSpec(
        name="grad_ratio",
        access=(UNLEARNED, FINETUNED, BOTH_CHECKPOINTS, LOGITS, LABELS, PER_EXAMPLE_LOSS, PER_EXAMPLE_GRADIENTS),
        backward_unlearned=1,
        backward_finetuned=1,
        models_resident=("unlearned", "finetuned"),
        note="calibrated gradient magnitude: two backward passes, two checkpoints",
    ),
    "log_grad_ratio": DetectorSpec(
        name="log_grad_ratio",
        access=(UNLEARNED, FINETUNED, BOTH_CHECKPOINTS, LOGITS, LABELS, PER_EXAMPLE_LOSS, PER_EXAMPLE_GRADIENTS),
        backward_unlearned=1,
        backward_finetuned=1,
        models_resident=("unlearned", "finetuned"),
        note="log form of grad_ratio; identical cost and identical ranking",
    ),
    "raw_cosine": DetectorSpec(
        name="raw_cosine",
        access=(UNLEARNED, FINETUNED, BOTH_CHECKPOINTS, LOGITS, LABELS,
                PER_EXAMPLE_LOSS, PER_EXAMPLE_GRADIENTS, DISPLACEMENT),
        backward_unlearned=1,
        needs_direction=True,
        models_resident=("unlearned", "finetuned"),
        note=(
            "signed alignment with ghat. The displacement is a one-off "
            "subtraction of two checkpoints, so its per-sample cost is zero, but "
            "it requires holding both."
        ),
    ),
    "alignment_cos_sq": DetectorSpec(
        name="alignment_cos_sq",
        access=(UNLEARNED, FINETUNED, BOTH_CHECKPOINTS, LOGITS, LABELS,
                PER_EXAMPLE_LOSS, PER_EXAMPLE_GRADIENTS, DISPLACEMENT),
        backward_unlearned=1,
        needs_direction=True,
        models_resident=("unlearned", "finetuned"),
        note="sign-invariant form of raw_cosine; identical cost",
    ),
    "sif": DetectorSpec(
        name="sif",
        access=(UNLEARNED, LOGITS, LABELS, PER_EXAMPLE_LOSS, PER_EXAMPLE_GRADIENTS, CURVATURE,
                AUXILIARY_DATA),
        backward_unlearned=1,
        needs_curvature=True,
        models_resident=("unlearned",),
        threat_band=INSIDER,
        note=(
            "adds a held-out curvature slice: auxiliary data the auditor must "
            "possess AND must keep disjoint from what it scores. At Fisher rank "
            "200 the curvature term was inert in the exploratory runs and SIF "
            "tracked grad_norm_sq at rho = 0.9999."
        ),
    ),
    "combined_log_score": DetectorSpec(
        name="combined_log_score",
        access=(UNLEARNED, FINETUNED, BOTH_CHECKPOINTS, LOGITS, LABELS,
                PER_EXAMPLE_LOSS, PER_EXAMPLE_GRADIENTS, DISPLACEMENT, CURVATURE, AUXILIARY_DATA),
        backward_unlearned=1,
        needs_direction=True,
        needs_curvature=True,
        models_resident=("unlearned", "finetuned"),
        note="the original combined score; the widest access set in the study",
    ),
    "fuma_lora_ratio": DetectorSpec(
        name="fuma_lora_ratio",
        access=(UNLEARNED, LOGITS, LABELS, PER_EXAMPLE_LOSS, PER_EXAMPLE_GRADIENTS),
        backward_shadow=1,
        models_resident=("shadow_bf16",),
        threat_band=INSIDER,
        note=(
            "needs an UNQUANTISED copy of the model: bitsandbytes does not "
            "backpropagate into 4-bit weights, so the backbone denominator does "
            "not exist on the model that was actually unlearned. That copy is "
            "the most expensive single object in the benchmark."
        ),
    ),
}

# No detector in this study uses a trained attack classifier. Recorded as an
# explicit zero rather than an omission: LiRA-style attacks do, and a reader
# comparing against them needs to see that this axis was considered.
UNUSED_ACCESS_DIMENSIONS = (ATTACK_CLASSIFIER,)


def comparable(challenger: str, baseline: str) -> bool:
    """True when ``challenger`` may be claimed as advantageous over ``baseline``.

    The rule: the baseline's access must be a SUBSET of the challenger's. Beating
    a detector that needs strictly more information than you do is a result;
    beating one that needs less is not, and beating one whose requirements are
    merely different is not comparable at all.
    """
    if challenger not in ACCESS or baseline not in ACCESS:
        raise KeyError(
            f"Unknown detector in comparison {challenger!r} vs {baseline!r}; "
            f"known detectors are {sorted(ACCESS)}"
        )
    return set(ACCESS[baseline].access) <= set(ACCESS[challenger].access)


def access_table() -> List[Dict[str, Any]]:
    """The access matrix, one row per detector, for the paper's Table 3."""
    return [
        {
            "detector": spec.name,
            "access": list(spec.access),
            "n_access_dimensions": len(spec.access),
            "forward_passes_per_sample": spec.forward_passes_per_sample,
            "backward_passes_per_sample": spec.backward_passes_per_sample,
            "models_resident": list(spec.models_resident),
            "needs_displacement": spec.needs_direction,
            "needs_curvature": spec.needs_curvature,
            "threat_band": spec.threat_band,
            "comparable_to_matched_access_baseline": comparable(
                spec.name, "loss_ratio_vs_finetuned"
            ),
            "note": spec.note,
        }
        for spec in sorted(ACCESS.values(), key=lambda s: s.name)
    ]


# --------------------------------------------------------------- measurement


@dataclass
class Timing:
    """Repeated timings of one primitive, summarised the way the brief asks."""

    name: str
    samples: List[float] = field(default_factory=list)
    unit: str = "seconds"
    n_items: int = 1
    peak_gpu_bytes: Optional[int] = None
    detail: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        if not self.samples:
            return {"name": self.name, "n_repeats": 0, "median": None,
                    "min": None, "max": None, "unit": self.unit}
        return {
            "name": self.name,
            "n_repeats": len(self.samples),
            # Median, not mean: a single scheduling hiccup on a shared GPU moves
            # a mean of five and does not move a median.
            "median": statistics.median(self.samples),
            "min": min(self.samples),
            "max": max(self.samples),
            "unit": self.unit,
            "n_items": self.n_items,
            "per_item_median": statistics.median(self.samples) / max(self.n_items, 1),
            "peak_gpu_bytes": self.peak_gpu_bytes,
            "peak_gpu_mib": (
                None if self.peak_gpu_bytes is None
                else round(self.peak_gpu_bytes / (1024 ** 2), 1)
            ),
            **self.detail,
        }


def gpu_processes() -> List[Dict[str, str]]:
    """Other processes holding GPU memory, recorded beside every timing.

    A benchmark on a shared card is only interpretable if what else was on the
    card is on the record. This does not correct for contention -- it documents
    it, so a reader can see whether the numbers were taken on an idle GPU.
    """
    try:
        output = subprocess.run(
            ["nvidia-smi",
             "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            rows.append({"pid": parts[0], "name": parts[1], "used_memory": parts[2]})
    return rows


def peak_host_bytes() -> Optional[int]:
    """Peak resident set size of this process, if the platform reports it."""
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes; macOS reports bytes.
    return int(usage) * (1024 if platform.system() == "Linux" else 1)


def _reset_gpu_peak() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _gpu_peak() -> Optional[int]:
    import torch

    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def time_it(fn, repeats: int, warmup: int, name: str, n_items: int = 1) -> Timing:
    """Run ``fn`` ``warmup`` times untimed, then ``repeats`` times timed.

    The warmup is not optional politeness: the first call through a CUDA kernel
    pays autotuning and allocator costs that can exceed the steady-state time
    several times over, and including it would make every detector's first
    measurement its worst.
    """
    import torch

    for _ in range(max(0, warmup)):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    _reset_gpu_peak()
    samples: List[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)

    return Timing(name=name, samples=samples, n_items=n_items,
                  peak_gpu_bytes=_gpu_peak())


def measure_primitives(
    config: Mapping[str, Any],
    unlearned_checkpoint: str,
    finetuned_checkpoint: str,
    manifest_path: Optional[str] = None,
    seed: int = 42,
    n_samples: int = 32,
    repeats: int = 5,
    warmup: int = 2,
    include_shadow: bool = True,
) -> Dict[str, Any]:
    """Measure every primitive the composed costs are built from.

    Deliberately small -- the per-sample cost is what gets composed, so a
    32-sample batch repeated five times gives a stable median without spending
    an hour to re-derive a number that scales linearly.
    """
    import torch

    from .data import collate_fn, load_or_build_splits, subsample
    from .gradients import (
        ParamScope,
        accumulate_gradient_stats,
        collect_named_grads,
        parameter_difference,
        select_param_names,
    )
    from .models import free, load_base_model, load_checkpoint_auto, load_shadow_model
    from .reproducibility import set_seed

    set_seed(seed)
    device = torch.device(config["device"])
    max_length = int(config["scoring"].get("max_length", config["data"]["max_length"]))

    split_info = load_or_build_splits(config, seed, manifest_path)
    splits = subsample(split_info["splits"], {"forget": n_samples, "retain": 0, "test": 0})
    samples = list(splits["forget"])[:n_samples]
    if not samples:
        raise ValueError("No samples available to benchmark")

    results: Dict[str, Any] = {
        "n_samples": len(samples),
        "repeats": repeats,
        "warmup": warmup,
        "max_length": max_length,
        "batch_size": 1,
        "sequence_length_policy": (
            f"per-sample, truncated at {max_length}; identical to the scoring "
            f"configuration, so composed costs transfer to a real run"
        ),
        "gpu_processes_at_start": gpu_processes(),
        "timings": {},
    }
    timings: Dict[str, Any] = results["timings"]

    # ---- phase 1: checkpoint loading -----------------------------------
    _reset_gpu_peak()
    start = time.perf_counter()
    base_model, tokenizer = load_base_model(config)
    timings["load_pretrained_base"] = Timing(
        "load_pretrained_base", [time.perf_counter() - start], peak_gpu_bytes=_gpu_peak()
    ).as_dict()

    _reset_gpu_peak()
    start = time.perf_counter()
    unlearned, _ = load_checkpoint_auto(
        config, unlearned_checkpoint, is_trainable=True, base_model=base_model
    )
    timings["load_unlearned_adapter"] = Timing(
        "load_unlearned_adapter", [time.perf_counter() - start], peak_gpu_bytes=_gpu_peak()
    ).as_dict()

    _reset_gpu_peak()
    start = time.perf_counter()
    finetuned_base, _ = load_base_model(config)
    finetuned, _ = load_checkpoint_auto(
        config, finetuned_checkpoint, is_trainable=True, base_model=finetuned_base
    )
    timings["load_finetuned_checkpoint"] = Timing(
        "load_finetuned_checkpoint", [time.perf_counter() - start],
        peak_gpu_bytes=_gpu_peak(),
    ).as_dict()

    # ---- phase 2: preprocessing ----------------------------------------
    def tokenize_all():
        for sample in samples:
            collate_fn(
                [{"text": sample.text, "sample_id": sample.sample_id}],
                tokenizer=tokenizer, max_length=max_length,
            )

    timings["preprocess"] = time_it(
        tokenize_all, repeats, warmup, "preprocess", n_items=len(samples)
    ).as_dict()

    batches = [
        {
            k: v.to(device)
            for k, v in collate_fn(
                [{"text": s.text, "sample_id": s.sample_id}],
                tokenizer=tokenizer, max_length=max_length,
            ).items()
        }
        for s in samples
    ]

    # ---- phase 3: per-example scoring primitives ------------------------
    param_names = select_param_names(
        unlearned, ParamScope(config["scoring"]["param_scope"]),
        config["scoring"].get("param_name_patterns"),
    )

    def forward_only(model):
        def run():
            with torch.no_grad():
                for batch in batches:
                    model(**batch)
        return run

    def forward_backward(model, names):
        # The gradient reduction is included in the timing on purpose: it is
        # part of what a gradient detector costs, it runs in float64 layer by
        # layer, and leaving it out would make the backward family look cheaper
        # than the collector actually finds it to be.
        def run():
            for batch in batches:
                model.zero_grad(set_to_none=True)
                loss = model(**batch).loss
                loss.backward()
                accumulate_gradient_stats(collect_named_grads(model, names))
        return run

    unlearned.eval()
    timings["forward_unlearned"] = time_it(
        forward_only(unlearned), repeats, warmup, "forward_unlearned",
        n_items=len(samples),
    ).as_dict()
    timings["forward_finetuned"] = time_it(
        forward_only(finetuned), repeats, warmup, "forward_finetuned",
        n_items=len(samples),
    ).as_dict()
    timings["forward_base"] = time_it(
        forward_only(base_model), repeats, warmup, "forward_base",
        n_items=len(samples),
    ).as_dict()
    timings["backward_unlearned"] = time_it(
        forward_backward(unlearned, param_names), repeats, warmup,
        "backward_unlearned", n_items=len(samples),
    ).as_dict()

    finetuned_names = select_param_names(
        finetuned, ParamScope(config["scoring"]["param_scope"]),
        config["scoring"].get("param_name_patterns"),
    )
    timings["backward_finetuned"] = time_it(
        forward_backward(finetuned, finetuned_names), repeats, warmup,
        "backward_finetuned", n_items=len(samples),
    ).as_dict()

    # ---- one-off costs --------------------------------------------------
    _reset_gpu_peak()
    start = time.perf_counter()
    direction = parameter_difference(unlearned, finetuned, param_names)
    timings["build_direction_once"] = Timing(
        "build_direction_once", [time.perf_counter() - start],
        peak_gpu_bytes=_gpu_peak(),
        detail={"n_parameters": sum(t.numel() for t in direction.values())},
    ).as_dict()
    del direction

    free(finetuned, finetuned_base)

    if include_shadow and not bool(config.get("training_mode") == "full"):
        try:
            _reset_gpu_peak()
            start = time.perf_counter()
            shadow = load_shadow_model(config, unlearned_checkpoint)
            shadow.eval()
            timings["load_shadow_bf16"] = Timing(
                "load_shadow_bf16", [time.perf_counter() - start],
                peak_gpu_bytes=_gpu_peak(),
            ).as_dict()
            shadow_names = [n for n, p in shadow.named_parameters() if p.requires_grad]
            timings["backward_shadow"] = time_it(
                forward_backward(shadow, shadow_names), repeats, warmup,
                "backward_shadow", n_items=len(samples),
            ).as_dict()
            free(shadow)
        except (RuntimeError, ValueError) as exc:
            timings["load_shadow_bf16"] = {"name": "load_shadow_bf16",
                                           "unavailable": str(exc)}
            timings["backward_shadow"] = {"name": "backward_shadow",
                                          "unavailable": str(exc)}

    free(unlearned, base_model)

    results["peak_host_bytes"] = peak_host_bytes()
    results["gpu_processes_at_end"] = gpu_processes()
    return results


# --------------------------------------------------------------- composition


def _per_item(timings: Mapping[str, Any], key: str) -> Optional[float]:
    entry = timings.get(key) or {}
    value = entry.get("per_item_median")
    return None if value is None else float(value)


def _one_off(timings: Mapping[str, Any], key: str) -> Optional[float]:
    entry = timings.get(key) or {}
    value = entry.get("median")
    return None if value is None else float(value)


def compose_costs(
    primitives: Mapping[str, Any],
    n_samples: int,
    curvature_rank: int = 200,
) -> List[Dict[str, Any]]:
    """Per-detector standalone and marginal cost, composed from the primitives.

    ``standalone`` is what the detector costs if it is the only thing computed:
    every model it needs is loaded and every pass it needs is paid for.

    ``marginal`` is what it ADDS to a run that already computes the shared
    backward pass on the unlearned model -- which is what the collector actually
    does, and is why four of these detectors are effectively free once any one of
    them is being computed.

    Reporting only standalone would overstate the gradient family's cost in this
    pipeline; reporting only marginal would understate what an auditor pays to
    obtain a gradient detector at all. Both are in the table.
    """
    timings = primitives.get("timings", {})
    forward_unlearned = _per_item(timings, "forward_unlearned")
    forward_finetuned = _per_item(timings, "forward_finetuned")
    forward_base = _per_item(timings, "forward_base")
    backward_unlearned = _per_item(timings, "backward_unlearned")
    backward_finetuned = _per_item(timings, "backward_finetuned")
    backward_shadow = _per_item(timings, "backward_shadow")

    load = {
        "unlearned": (_one_off(timings, "load_pretrained_base") or 0.0)
        + (_one_off(timings, "load_unlearned_adapter") or 0.0),
        "finetuned": _one_off(timings, "load_finetuned_checkpoint"),
        "base": _one_off(timings, "load_pretrained_base"),
        "shadow_bf16": _one_off(timings, "load_shadow_bf16"),
    }
    peak = {
        "unlearned": (timings.get("backward_unlearned") or {}).get("peak_gpu_mib"),
        "finetuned": (timings.get("backward_finetuned") or {}).get("peak_gpu_mib"),
        "base": (timings.get("forward_base") or {}).get("peak_gpu_mib"),
        "shadow_bf16": (timings.get("backward_shadow") or {}).get("peak_gpu_mib"),
    }

    rows: List[Dict[str, Any]] = []
    for spec in sorted(ACCESS.values(), key=lambda s: s.name):
        per_sample_terms = [
            (spec.forward_unlearned, forward_unlearned),
            (spec.forward_finetuned, forward_finetuned),
            (spec.forward_base, forward_base),
            (spec.backward_unlearned, backward_unlearned),
            (spec.backward_finetuned, backward_finetuned),
            (spec.backward_shadow, backward_shadow),
        ]
        known = [count * value for count, value in per_sample_terms
                 if count and value is not None]
        unmeasured = [count for count, value in per_sample_terms
                      if count and value is None]
        per_sample = sum(known) if known and not unmeasured else None

        # Marginal: the unlearned backward is already being paid for by the
        # shared collection pass, so a detector that needs only that adds
        # nothing per sample.
        marginal_terms = [
            (spec.forward_finetuned, forward_finetuned),
            (spec.forward_base, forward_base),
            (spec.backward_finetuned, backward_finetuned),
            (spec.backward_shadow, backward_shadow),
        ]
        marginal_known = [count * value for count, value in marginal_terms
                          if count and value is not None]
        marginal_unmeasured = [count for count, value in marginal_terms
                               if count and value is None]
        marginal_per_sample = (
            sum(marginal_known) if not marginal_unmeasured else None
        )
        # A forward-only pass on the unlearned model contributes nothing to the
        # marginal cost: the shared collection pass already ran a backward
        # through that model, and the loss falls out of it for free. That is why
        # forward_unlearned is absent from `marginal_terms` above.

        one_off = 0.0
        one_off_terms: List[str] = []
        for model in spec.models_resident:
            value = load.get(model)
            if value is not None:
                one_off += value
                one_off_terms.append(f"load {model}")
        if spec.needs_direction:
            value = _one_off(timings, "build_direction_once")
            if value is not None:
                one_off += value
                one_off_terms.append("build displacement")
        if spec.needs_curvature and backward_unlearned is not None:
            curvature_cost = curvature_rank * backward_unlearned
            one_off += curvature_cost
            one_off_terms.append(
                f"Fisher curvature over {curvature_rank} held-out samples"
            )

        rows.append(
            {
                "detector": spec.name,
                "access": list(spec.access),
                "threat_band": spec.threat_band,
                "forward_passes_per_sample": spec.forward_passes_per_sample,
                "backward_passes_per_sample": spec.backward_passes_per_sample,
                "models_resident": list(spec.models_resident),
                "standalone_seconds_per_sample": per_sample,
                "marginal_seconds_per_sample": marginal_per_sample,
                "one_off_seconds": one_off,
                "one_off_terms": one_off_terms,
                "standalone_seconds_total": (
                    None if per_sample is None else per_sample * n_samples + one_off
                ),
                "throughput_samples_per_second": (
                    None if not per_sample else 1.0 / per_sample
                ),
                "peak_gpu_mib": max(
                    [peak.get(m) for m in spec.models_resident if peak.get(m) is not None]
                    or [None],
                    key=lambda v: (v is not None, v),
                ),
                "persistent_storage": _storage_note(spec),
                "comparable_to_matched_access_baseline": comparable(
                    spec.name, "loss_ratio_vs_finetuned"
                ),
                "note": spec.note,
                "unmeasured_terms": bool(unmeasured),
            }
        )
    return rows


def _storage_note(spec: DetectorSpec) -> str:
    """What the detector's holder must keep on disk, in words rather than bytes."""
    needs = []
    if "unlearned" in spec.models_resident or spec.backward_shadow:
        needs.append("the published (post-deletion) checkpoint")
    if "finetuned" in spec.models_resident:
        needs.append("the pre-deletion checkpoint, retained after deletion")
    if "base" in spec.models_resident:
        needs.append("the pretrained base (public)")
    if spec.needs_curvature:
        needs.append("a held-out curvature slice and its Fisher factors")
    return "; ".join(needs) if needs else "none"


# -------------------------------------------------------------------- the CLI


def build_results(
    primitives: Optional[Mapping[str, Any]],
    n_samples: int,
    curvature_rank: int = 200,
) -> Dict[str, Any]:
    """Assemble Component C's result document."""
    document: Dict[str, Any] = {
        "component": "C -- matched-access detector cost and capability",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "matched_access_baseline": "loss_ratio_vs_finetuned",
        "matched_access_rule": (
            "a challenger may be claimed advantageous over a baseline only when "
            "the baseline's access is a subset of the challenger's"
        ),
        "unused_access_dimensions": list(UNUSED_ACCESS_DIMENSIONS),
        "access_table": access_table(),
        "central_comparison": {
            "baseline": "loss_ratio_vs_finetuned",
            "challengers": [
                "raw_cosine", "alignment_cos_sq", "grad_ratio", "sif",
                "combined_log_score",
            ],
            "all_challengers_comparable": all(
                comparable(c, "loss_ratio_vs_finetuned")
                for c in ("raw_cosine", "alignment_cos_sq", "grad_ratio",
                          "combined_log_score")
            ),
            "note": (
                "sif and grad_norm_sq do NOT subsume the baseline's access: they "
                "never read the pre-deletion checkpoint. They are reported for "
                "completeness and no advantage over the baseline may be claimed "
                "for them."
            ),
        },
    }
    if primitives is not None:
        document["primitives"] = primitives
        document["costs"] = compose_costs(primitives, n_samples, curvature_rank)
        document["environment"] = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "gpu_processes": primitives.get("gpu_processes_at_start", []),
            "peak_host_bytes": primitives.get("peak_host_bytes"),
        }
    return document


COST_CSV_COLUMNS = (
    "detector", "threat_band", "forward_per_sample", "backward_per_sample",
    "models_resident", "standalone_s_per_sample", "marginal_s_per_sample",
    "throughput_per_s", "one_off_s", "peak_gpu_mib", "comparable_to_baseline",
)


def write_csv(document: Mapping[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows = document.get("costs") or [
        {
            "detector": row["detector"],
            "threat_band": row["threat_band"],
            "forward_passes_per_sample": row["forward_passes_per_sample"],
            "backward_passes_per_sample": row["backward_passes_per_sample"],
            "models_resident": row["models_resident"],
            "comparable_to_matched_access_baseline": row[
                "comparable_to_matched_access_baseline"
            ],
        }
        for row in document["access_table"]
    ]

    def number(value: Any, digits: int = 6) -> str:
        if value is None:
            return ""
        try:
            as_float = float(value)
        except (TypeError, ValueError):
            return ""
        return "" if not math.isfinite(as_float) else f"{as_float:.{digits}f}"

    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COST_CSV_COLUMNS)
        for row in rows:
            writer.writerow(
                [
                    row["detector"],
                    row.get("threat_band", ""),
                    row["forward_passes_per_sample"],
                    row["backward_passes_per_sample"],
                    "+".join(row.get("models_resident", [])),
                    number(row.get("standalone_seconds_per_sample")),
                    number(row.get("marginal_seconds_per_sample")),
                    number(row.get("throughput_samples_per_second"), 2),
                    number(row.get("one_off_seconds"), 2),
                    row.get("peak_gpu_mib", "") or "",
                    row.get("comparable_to_matched_access_baseline", ""),
                ]
            )
    return path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detector access classification, and a runtime/memory benchmark "
            "composed from measured primitives."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # access table only -- no GPU, no models loaded\n"
            "  $PY -m deletion_signals.benchmark --access-only \\\n"
            "      --output fllm_audit/benchmark\n\n"
            "  # full benchmark on the real checkpoints\n"
            "  $PY -m deletion_signals.benchmark --config arxivmia_confirm.yaml \\\n"
            "      --unlearned-checkpoint ds_confirm/controls/exact_retrain/seed42/model \\\n"
            "      --finetuned-checkpoint ds_checkpoints/finetuned_seed42 \\\n"
            "      --manifest results/seed42/splits.json \\\n"
            "      --output fllm_audit/benchmark\n"
        ),
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--access-only", action="store_true",
                        help="Emit the access table without loading any model")
    parser.add_argument("--unlearned-checkpoint", default=None)
    parser.add_argument("--finetuned-checkpoint", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--scale-to", type=int, default=800,
                        help="Sample count the composed totals are reported for "
                             "(default: the 200 forget + 600 retain a real run scores)")
    parser.add_argument("--no-shadow", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="./fllm_audit/benchmark")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    primitives = None
    if not args.access_only:
        from .config import apply_cli_overrides, load_config

        if not (args.unlearned_checkpoint and args.finetuned_checkpoint):
            raise SystemExit(
                "--unlearned-checkpoint and --finetuned-checkpoint are required "
                "unless --access-only is given"
            )
        config = load_config(args.config)
        if args.device:
            config["device"] = args.device
        config = apply_cli_overrides(config, args.overrides)
        primitives = measure_primitives(
            config,
            unlearned_checkpoint=args.unlearned_checkpoint,
            finetuned_checkpoint=args.finetuned_checkpoint,
            manifest_path=args.manifest,
            seed=args.seed,
            n_samples=args.n_samples,
            repeats=args.repeats,
            warmup=args.warmup,
            include_shadow=not args.no_shadow,
        )

    document = build_results(primitives, args.scale_to)

    os.makedirs(args.output, exist_ok=True)
    json_path = os.path.join(args.output, "benchmark.json")
    with open(json_path, "w") as handle:
        json.dump(document, handle, indent=2, allow_nan=False, default=str)
    csv_path = write_csv(document, os.path.join(args.output, "detector_costs.csv"))

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
