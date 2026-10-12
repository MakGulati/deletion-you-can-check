"""Per-sample score collection.

Takes a fine-tuned checkpoint, an unlearned checkpoint and a split manifest, and
writes one row per sample. It never asks how the unlearned checkpoint was
produced, which is what makes the evaluation method-agnostic: GA+GD, SCRUB, or
anything registered later all arrive here as a directory on disk.

Backward-pass budget, per sample
--------------------------------
=====================================  ========  ====================================
model                                  passes    what it yields
=====================================  ========  ====================================
unlearned                              1 bwd     loss, ||∇L||², cos, cos², SIF
pretrained base                        1 fwd     loss_base (calibrated loss)
fine-tuned pre-unlearning              1 bwd     ||∇L_ref||² (calibrated gradient)
fine-tuned pre-unlearning              1 fwd     loss_finetuned (loss_vs_finetuned)
bf16 shadow (LoRA mode, FUMA on)       1 bwd     both halves of the FUMA ratio
=====================================  ========  ====================================

The fine-tuned model appears twice because it answers two questions, but it is
loaded once; and when the calibrated loss already uses it as its reference, the
extra forward pass is skipped rather than repeated.

Every unlearned-model gradient statistic comes out of that single backward pass;
nothing is recomputed. Full gradient vectors are never stored -- only the scalar
reductions -- except transiently for the optional Fisher self-influence
ablation, which needs the vector by construction.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from . import SCHEMA_VERSION
from .config import (
    REFERENCE_BASE,
    REFERENCE_FINETUNED,
    REFERENCE_NONE,
    apply_cli_overrides,
    effective_config,
    is_full_finetuning,
    load_config,
)
from .data import Sample, load_or_build_splits, subsample
from .gradients import (
    FlatGradientBuffer,
    ParamScope,
    accumulate_gradient_stats,
    collect_named_grads,
    direction_norm,
    parameter_difference,
    parameter_numels,
    select_param_names,
)
from .models import (
    describe_model,
    free,
    load_base_model,
    load_checkpoint_auto,
    load_shadow_model,
)
from .orientations import OrientationTable
from .reproducibility import run_provenance, set_seed
from .schema import POPULATIONS, ScoreRow, make_flags, write_score_file
from .scores import (
    FumaNorm,
    FumaWeighting,
    cos_sq,
    combined_log_score,
    fuma_lora_ratio,
    grad_ratio,
    log_grad_ratio,
    loss_delta,
    loss_ratio,
)
from .sif import FisherSelfInfluence, gradient_self_influence


@dataclass
class ModelBundle:
    """The models a collection run needs, and what each is for."""

    unlearned: Any
    tokenizer: Any
    finetuned: Optional[Any] = None
    base: Optional[Any] = None
    shadow: Optional[Any] = None
    descriptions: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.descriptions is None:
            self.descriptions = {}


class ScoreCollector:
    """Computes every enabled score for a population of samples."""

    def __init__(
        self,
        config: Mapping[str, Any],
        bundle: ModelBundle,
        device: torch.device,
    ):
        self.config = config
        self.bundle = bundle
        self.device = device

        scoring = config["scoring"]
        self.eps = float(scoring["eps"])
        self.max_length = int(scoring.get("max_length", config["data"]["max_length"]))
        self.scope = ParamScope(scoring["param_scope"])
        self.name_patterns = scoring.get("param_name_patterns")

        self.param_names = select_param_names(
            bundle.unlearned, self.scope, self.name_patterns
        )
        self.param_numels = parameter_numels(bundle.unlearned, self.param_names)

        scores = config["scores"]
        self.want_loss = scores["loss_unlearned"]["enabled"]
        self.loss_reference = (
            scores["loss_calibrated"]["reference"]
            if scores["loss_calibrated"]["enabled"]
            else REFERENCE_NONE
        )
        # The second, independent loss reference. It needs no extra model: the
        # fine-tuned checkpoint is already resident for ghat and for the
        # gradient calibration.
        self.want_loss_vs_finetuned = bool(
            scores.get("loss_vs_finetuned", {}).get("enabled", False)
            and bundle.finetuned is not None
        )
        self.want_grad_norm = scores["grad_norm_sq"]["enabled"]
        self.grad_reference = (
            scores["grad_calibrated"]["reference"]
            if scores["grad_calibrated"]["enabled"]
            else REFERENCE_NONE
        )
        self.fuma_cfg = scores["fuma"]
        self.sif_cfg = scores["sif"]

        # ĝ = θ_unlearned − θ_finetuned, held per parameter so the layerwise
        # accumulation can consume it without a concatenated d-vector.
        self.direction: Optional[Dict[str, torch.Tensor]] = None
        self.direction_norm: float = float("nan")
        if bundle.finetuned is not None:
            self.direction = parameter_difference(
                bundle.unlearned, bundle.finetuned, self.param_names
            )
            self.direction_norm = direction_norm(self.direction)

        # The reference model whose gradients calibrate the magnitude.
        self.grad_ref_model = None
        self.grad_ref_names: List[str] = []
        if self.grad_reference == REFERENCE_FINETUNED and bundle.finetuned is not None:
            self.grad_ref_model = bundle.finetuned
            self.grad_ref_names = select_param_names(
                bundle.finetuned, self.scope, self.name_patterns
            )
        elif self.grad_reference == REFERENCE_BASE and bundle.base is not None:
            self.grad_ref_model = bundle.base
            self.grad_ref_names = select_param_names(
                bundle.base, self.scope, self.name_patterns
            )
        # Precomputed once: parameter_numels() walks every named parameter, and
        # calling it per sample on a 1B model costs more than the backward pass
        # it accompanies.
        self.grad_ref_numels = (
            parameter_numels(self.grad_ref_model, self.grad_ref_names)
            if self.grad_ref_model is not None
            else {}
        )

        # The model FUMA's two magnitudes come from.
        self.fuma_model = None
        self.fuma_names: List[str] = []
        self.fuma_numels: Dict[str, int] = {}
        self.fuma_na_reason: Optional[str] = None
        self.fuma_source: Optional[str] = None
        self.fuma_unfroze_backbone = False
        self._configure_fuma()
        if self.fuma_model is not None:
            self.fuma_numels = parameter_numels(self.fuma_model, self.fuma_names)

        # Optional self-influence ablation.
        self.fisher: Optional[FisherSelfInfluence] = None
        self.flat_buffer: Optional[FlatGradientBuffer] = None
        if self.sif_cfg["enabled"] and self.sif_cfg["method"] == "fisher":
            self.fisher = FisherSelfInfluence(
                damping=float(self.sif_cfg["damping"]),
                rank=int(self.sif_cfg["fisher_rank"]),
                device=str(self.sif_cfg["fisher_device"]),
            )
            self.flat_buffer = FlatGradientBuffer.for_model(
                bundle.unlearned, self.param_names
            )

    # ------------------------------------------------------------------ setup

    def _configure_fuma(self) -> None:
        if not self.fuma_cfg["enabled"]:
            self.fuma_na_reason = "disabled in configuration (scores.fuma.enabled=false)"
            return

        source = self.fuma_cfg["denominator_source"]
        self.fuma_source = source

        if source == "none":
            self.fuma_na_reason = (
                "scores.fuma.denominator_source='none': the non-LoRA gradient "
                "magnitude was not obtained, so the ratio is not computed"
            )
            return

        if source == "shadow_bf16":
            if self.bundle.shadow is None:
                self.fuma_na_reason = (
                    "the bf16 shadow model required for the FUMA denominator was "
                    "not loaded (see the run log for why)"
                )
                return
            self.fuma_model = self.bundle.shadow
            self.fuma_names = select_param_names(self.bundle.shadow, ParamScope.FULL)
            return

        # same_model: both magnitudes come from the scored model itself. That
        # only works if its non-LoRA parameters actually receive gradients,
        # which under PEFT they do not -- the backbone is frozen. Unfreeze it
        # here, after self.param_names has already been fixed, so the scored
        # parameter space is unchanged and only the FUMA denominator gains
        # anything. Recorded as a discrepancy, because the backbone was frozen
        # during training and these gradients describe a direction the
        # optimiser never took.
        model = self.bundle.unlearned
        frozen_non_lora = [
            param
            for name, param in model.named_parameters()
            if "lora" not in name.lower() and not param.requires_grad
        ]
        if frozen_non_lora:
            if any(
                "4bit" in str(param.dtype) or "uint8" in str(param.dtype)
                for param in frozen_non_lora
            ):
                self.fuma_na_reason = (
                    "scores.fuma.denominator_source='same_model' but the backbone "
                    "is 4-bit quantised; bitsandbytes does not backpropagate into "
                    "4-bit weights. Use denominator_source='shadow_bf16'."
                )
                return
            for param in frozen_non_lora:
                param.requires_grad = True
            self.fuma_unfroze_backbone = True

        self.fuma_model = model
        self.fuma_names = select_param_names(model, ParamScope.FULL)

    # ---------------------------------------------------------------- helpers

    def _tokenize(self, text: str) -> Tuple[Dict[str, torch.Tensor], bool]:
        encoded = self.bundle.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        truncated = int(encoded["input_ids"].shape[1]) >= self.max_length
        return {k: v.to(self.device) for k, v in encoded.items()}, truncated

    @staticmethod
    def _forward(model, inputs: Mapping[str, torch.Tensor]):
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["input_ids"],
        )

    def _forward_backward(
        self, model, names: Sequence[str], inputs: Mapping[str, torch.Tensor]
    ) -> Tuple[float, List[Tuple[str, Optional[torch.Tensor]]]]:
        """One backward pass; returns the loss and the requested gradients."""
        model.zero_grad(set_to_none=True)
        outputs = self._forward(model, inputs)
        outputs.loss.backward()
        return float(outputs.loss.item()), collect_named_grads(model, names)

    @torch.no_grad()
    def _loss_only(self, model, inputs: Mapping[str, torch.Tensor]) -> float:
        return float(self._forward(model, inputs).loss.item())

    # ----------------------------------------------------------------- scoring

    def build_fisher(self, curvature_samples: Sequence[Sample]) -> Optional[Dict[str, Any]]:
        """Populate the Fisher curvature estimate from a held-out retain slice."""
        if self.fisher is None:
            return None

        gradients = []
        fingerprints = []
        for sample in tqdm(curvature_samples, desc="Fisher curvature holdout"):
            inputs, _ = self._tokenize(sample.text)
            _, named_grads = self._forward_backward(
                self.bundle.unlearned, self.param_names, inputs
            )
            gradients.append(self.flat_buffer.flatten(named_grads, device=self.fisher.device))
            fingerprints.append(sample.text_sha1)

        self.fisher.build(gradients, fingerprints)
        return {
            "rank": len(gradients),
            "dim": self.flat_buffer.dim,
            "damping": self.fisher.damping,
            "device": str(self.fisher.device),
            "bytes": self.fisher.memory_bytes(),
        }

    def score_sample(self, sample: Sample, meta: Mapping[str, Any]) -> ScoreRow:
        """Compute every enabled score for one sample."""
        inputs, truncated = self._tokenize(sample.text)
        n_tokens = max(int(inputs["input_ids"].shape[1]) - 1, 0)

        # ---- the one backward pass on the unlearned model -------------------
        unlearned_loss, named_grads = self._forward_backward(
            self.bundle.unlearned, self.param_names, inputs
        )
        stats = accumulate_gradient_stats(
            named_grads, direction=self.direction, numel_hint=self.param_numels
        )

        grad_norm_sq = stats.norm_sq if self.want_grad_norm else None
        raw_cosine = stats.raw_cosine() if self.direction is not None else None
        alignment = cos_sq(raw_cosine)

        # ---- calibrated loss -------------------------------------------------
        base_loss: Optional[float] = None
        loss_reference_model: Optional[str] = None
        if self.loss_reference == REFERENCE_BASE and self.bundle.base is not None:
            base_loss = self._loss_only(self.bundle.base, inputs)
            loss_reference_model = f"pretrained_base:{self.config['model']['name']}"
        elif self.loss_reference == REFERENCE_FINETUNED and self.bundle.finetuned is not None:
            base_loss = self._loss_only(self.bundle.finetuned, inputs)
            loss_reference_model = f"finetuned:{meta.get('finetuned_checkpoint')}"

        # ---- loss against the pre-unlearning model ---------------------------
        # A different question from the block above: not "was this sample
        # fitted" but "what did unlearning do to it".
        finetuned_loss: Optional[float] = None
        loss_finetuned_model: Optional[str] = None
        if self.want_loss_vs_finetuned:
            if self.loss_reference == REFERENCE_FINETUNED and base_loss is not None:
                # The calibrated-loss block already ran this exact forward pass
                # on this exact model; a second one would return the same number.
                finetuned_loss = base_loss
            else:
                finetuned_loss = self._loss_only(self.bundle.finetuned, inputs)
            loss_finetuned_model = f"finetuned:{meta.get('finetuned_checkpoint')}"

        # ---- calibrated gradient magnitude -----------------------------------
        grad_norm_sq_ref: Optional[float] = None
        grad_reference_model: Optional[str] = None
        if self.grad_ref_model is not None:
            _, ref_grads = self._forward_backward(
                self.grad_ref_model, self.grad_ref_names, inputs
            )
            ref_stats = accumulate_gradient_stats(
                ref_grads, numel_hint=self.grad_ref_numels
            )
            grad_norm_sq_ref = ref_stats.norm_sq
            grad_reference_model = (
                f"finetuned:{meta.get('finetuned_checkpoint')}"
                if self.grad_reference == REFERENCE_FINETUNED
                else f"pretrained_base:{self.config['model']['name']}"
            )

        # ---- FUMA -------------------------------------------------------------
        fuma_value: Optional[float] = None
        fuma_reason: Optional[str] = self.fuma_na_reason
        if self.fuma_model is not None:
            if self.fuma_model is self.bundle.unlearned:
                # The backward above already populated .grad for every parameter
                # that requires one, so no second pass is needed.
                fuma_grads = collect_named_grads(self.fuma_model, self.fuma_names)
            else:
                _, fuma_grads = self._forward_backward(
                    self.fuma_model, self.fuma_names, inputs
                )
            fuma_stats = accumulate_gradient_stats(
                fuma_grads, numel_hint=self.fuma_numels
            )
            fuma_value, fuma_reason = fuma_lora_ratio(
                abs_sum_lora=fuma_stats.abs_sum_lora,
                abs_sum_non_lora=fuma_stats.abs_sum_non_lora,
                sq_sum_lora=fuma_stats.sq_sum_lora,
                sq_sum_non_lora=fuma_stats.sq_sum_non_lora,
                numel_lora=fuma_stats.numel_lora,
                numel_non_lora=fuma_stats.numel_non_lora,
                tensor_abs_mean_sum_lora=fuma_stats.tensor_abs_mean_sum_lora,
                tensor_abs_mean_sum_non_lora=fuma_stats.tensor_abs_mean_sum_non_lora,
                n_tensors_lora=fuma_stats.n_tensors_lora,
                n_tensors_non_lora=fuma_stats.n_tensors_non_lora,
                norm=FumaNorm(self.fuma_cfg["norm"]),
                weighting=FumaWeighting(self.fuma_cfg["weighting"]),
                eps=self.eps,
            )

        # ---- self-influence ablation -------------------------------------------
        sif_value: Optional[float] = None
        fisher_explained: Optional[float] = None
        if self.sif_cfg["enabled"] and grad_norm_sq is not None:
            if self.fisher is not None:
                flat = self.flat_buffer.flatten(named_grads, device=self.fisher.device)
                measurement = self.fisher.score(flat, grad_norm_sq)
            else:
                measurement = gradient_self_influence(
                    grad_norm_sq, float(self.sif_cfg["damping"])
                )
            sif_value = measurement.value
            fisher_explained = measurement.fisher_explained

        combined = (
            combined_log_score(sif_value, alignment, float(self.sif_cfg["beta"]))
            if self.sif_cfg["enabled"]
            else None
        )

        return ScoreRow(
            seed=int(meta["seed"]),
            method=str(meta["method"]),
            dataset=str(meta["dataset"]),
            model_id=str(meta["model_id"]),
            sample_id=sample.sample_id,
            text_sha1=sample.text_sha1,
            population=sample.population,
            unlearned_checkpoint=meta.get("unlearned_checkpoint"),
            finetuned_checkpoint=meta.get("finetuned_checkpoint"),
            n_tokens=n_tokens,
            loss_unlearned=unlearned_loss if self.want_loss else None,
            loss_base=base_loss,
            loss_reference_model=loss_reference_model,
            loss_ratio=loss_ratio(unlearned_loss, base_loss, self.eps),
            loss_delta=loss_delta(unlearned_loss, base_loss),
            loss_finetuned=finetuned_loss,
            loss_finetuned_model=loss_finetuned_model,
            loss_ratio_vs_finetuned=loss_ratio(unlearned_loss, finetuned_loss, self.eps),
            loss_delta_vs_finetuned=loss_delta(unlearned_loss, finetuned_loss),
            grad_norm_sq=grad_norm_sq,
            grad_norm_sq_ref=grad_norm_sq_ref,
            grad_reference_model=grad_reference_model,
            grad_ratio=(
                grad_ratio(grad_norm_sq, grad_norm_sq_ref, self.eps)
                if grad_norm_sq is not None
                else None
            ),
            log_grad_ratio=(
                log_grad_ratio(grad_norm_sq, grad_norm_sq_ref, self.eps)
                if grad_norm_sq is not None
                else None
            ),
            fuma_lora_ratio=fuma_value,
            fuma_na_reason=fuma_reason,
            fuma_source_model=self.fuma_source,
            raw_cosine=raw_cosine,
            alignment_cos_sq=alignment,
            sif=sif_value,
            fisher_explained=fisher_explained,
            combined_log_score=combined,
            flags=make_flags(
                zero_grad_norm=stats.norm_sq <= 0.0,
                zero_direction_norm=(
                    self.direction is not None and stats.direction_norm_sq <= 0.0
                ),
                nonfinite_grad=stats.n_nonfinite_tensors > 0
                or not math.isfinite(stats.norm_sq),
                nonfinite_loss=not math.isfinite(unlearned_loss),
                missing_grad_params=stats.n_missing_grad,
                truncated=truncated,
            ),
        )


# ---------------------------------------------------------------------- driver


def load_models(
    config: Mapping[str, Any],
    unlearned_checkpoint: str,
    finetuned_checkpoint: Optional[str],
) -> ModelBundle:
    """Load exactly the models the enabled scores require, and no others."""
    scores = config["scores"]
    device_needed_base = (
        scores["loss_calibrated"]["enabled"]
        and scores["loss_calibrated"]["reference"] == REFERENCE_BASE
    ) or (
        scores["grad_calibrated"]["enabled"]
        and scores["grad_calibrated"]["reference"] == REFERENCE_BASE
    )

    full_ft = is_full_finetuning(config)

    print(f"Loading unlearned checkpoint: {unlearned_checkpoint}")
    if full_ft:
        unlearned, tokenizer = load_checkpoint_auto(
            config, unlearned_checkpoint, is_trainable=True
        )
    else:
        backbone, tokenizer = load_base_model(config)
        unlearned, _ = load_checkpoint_auto(
            config, unlearned_checkpoint, is_trainable=True, base_model=backbone
        )
    unlearned.eval()  # disables dropout; gradients still flow

    finetuned = None
    if finetuned_checkpoint:
        print(f"Loading fine-tuned reference: {finetuned_checkpoint}")
        needs_grads = (
            scores["grad_calibrated"]["enabled"]
            and scores["grad_calibrated"]["reference"] == REFERENCE_FINETUNED
        )
        if full_ft:
            finetuned, _ = load_checkpoint_auto(
                config, finetuned_checkpoint, is_trainable=needs_grads
            )
        else:
            ft_backbone, _ = load_base_model(config)
            finetuned, _ = load_checkpoint_auto(
                config,
                finetuned_checkpoint,
                is_trainable=needs_grads,
                base_model=ft_backbone,
            )
        finetuned.eval()

    base = None
    if device_needed_base:
        print("Loading pretrained base model (difficulty reference)")
        needs_base_grads = (
            scores["grad_calibrated"]["enabled"]
            and scores["grad_calibrated"]["reference"] == REFERENCE_BASE
        )
        base, _ = load_base_model(config)
        for param in base.parameters():
            param.requires_grad = bool(needs_base_grads)
        base.eval()

    shadow = None
    if (
        scores["fuma"]["enabled"]
        and scores["fuma"]["denominator_source"] == "shadow_bf16"
        and not full_ft
    ):
        print("Loading bf16 shadow model (FUMA denominator)")
        try:
            shadow = load_shadow_model(config, unlearned_checkpoint)
            shadow.eval()
        except (RuntimeError, ValueError) as exc:
            print(f"  shadow model unavailable: {exc}")
            shadow = None

    descriptions = {"unlearned": describe_model(unlearned, unlearned_checkpoint)}
    if finetuned is not None:
        descriptions["finetuned"] = describe_model(finetuned, finetuned_checkpoint or "")
    if base is not None:
        descriptions["pretrained_base"] = describe_model(base, config["model"]["name"])
    if shadow is not None:
        descriptions["shadow_bf16"] = describe_model(shadow, unlearned_checkpoint)

    return ModelBundle(
        unlearned=unlearned,
        tokenizer=tokenizer,
        finetuned=finetuned,
        base=base,
        shadow=shadow,
        descriptions=descriptions,
    )


def collect_scores(
    config: Mapping[str, Any],
    seed: int,
    method: str,
    unlearned_checkpoint: str,
    finetuned_checkpoint: Optional[str],
    output_path: str,
    unlearning_run: Optional[Mapping[str, Any]] = None,
    manifest_path: Optional[str] = None,
) -> str:
    """Score every sample and write the per-sample table. Returns the path."""
    set_seed(seed)
    device = torch.device(config["device"])

    split_info = load_or_build_splits(config, seed, manifest_path)
    splits = subsample(
        split_info["splits"], config["scoring"].get("max_samples_per_population")
    )

    bundle = load_models(config, unlearned_checkpoint, finetuned_checkpoint)
    collector = ScoreCollector(config, bundle, device)

    if collector.direction is None:
        raise ValueError(
            "No fine-tuned checkpoint was supplied, so the unlearning direction "
            "ghat = theta_unlearned - theta_finetuned cannot be formed and the "
            "alignment score -- the whole point of this comparison -- is "
            "unavailable. Pass --finetuned-checkpoint."
        )

    # The Fisher curvature slice must be disjoint from every scored sample.
    fisher_info = None
    if collector.fisher is not None:
        rank = int(config["scores"]["sif"]["fisher_rank"])
        retain = splits["retain"]
        if len(retain) <= rank:
            raise ValueError(
                f"Need more than {rank} retain samples so the curvature slice can "
                f"be held out of scoring; got {len(retain)}. Lower "
                f"scores.sif.fisher_rank or widen the retain population."
            )
        curvature, splits["retain"] = retain[:rank], retain[rank:]
        fisher_info = collector.build_fisher(curvature)
        collector.fisher.assert_disjoint(
            sample.text_sha1
            for population in POPULATIONS
            for sample in splits.get(population, [])
        )

    meta_common = {
        "seed": seed,
        "method": method,
        "dataset": config["data"]["name"],
        "model_id": config["model"]["name"],
        "unlearned_checkpoint": os.path.abspath(unlearned_checkpoint),
        "finetuned_checkpoint": (
            os.path.abspath(finetuned_checkpoint) if finetuned_checkpoint else None
        ),
    }

    rows: List[ScoreRow] = []
    for population in POPULATIONS:
        samples = splits.get(population, [])
        for sample in tqdm(samples, desc=f"Scoring D_{population}"):
            rows.append(collector.score_sample(sample, meta_common))

    orientation_table = OrientationTable.load(
        config["evaluation"].get("orientations_path")
    )

    metadata: Dict[str, Any] = {
        **meta_common,
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "training_mode": config.get("training_mode", "lora"),
        "param_scope": config["scoring"]["param_scope"],
        "param_names_count": len(collector.param_names),
        "param_dim": sum(collector.param_numels.values()),
        "eps": collector.eps,
        "unlearning_direction": {
            "source": "param_diff",
            "definition": "ghat = theta_unlearned - theta_finetuned",
            "norm": collector.direction_norm,
            "access": (
                "white-box: both the fine-tuned pre-unlearning checkpoint and the "
                "unlearned checkpoint are read directly"
            ),
            "sign_note": (
                "cos^2 is sign-invariant, so this convention affects only "
                "raw_cosine, whose direction under it is preregistered in "
                "configs/orientations.yaml: higher cosine predicts 'forget' on "
                "every task"
            ),
        },
        "unlearning_run": dict(unlearning_run or {}),
        "references": {
            "loss_calibration": collector.loss_reference,
            "loss_vs_finetuned": collector.want_loss_vs_finetuned,
            "grad_calibration": collector.grad_reference,
        },
        "fuma": {
            "enabled": bool(config["scores"]["fuma"]["enabled"]),
            "denominator_source": collector.fuma_source,
            "norm": config["scores"]["fuma"]["norm"],
            "weighting": config["scores"]["fuma"]["weighting"],
            "applicable": collector.fuma_model is not None,
            "na_reason": collector.fuma_na_reason,
            "discrepancies": _fuma_discrepancies(config, collector),
        },
        "sif": {
            "enabled": bool(config["scores"]["sif"]["enabled"]),
            "method": config["scores"]["sif"]["method"],
            "beta": config["scores"]["sif"]["beta"],
            "fisher": fisher_info,
        },
        "populations": {p: len(splits.get(p, [])) for p in POPULATIONS},
        "split_source": split_info["source"],
        "split_manifest": split_info.get("manifest_path"),
        "models": bundle.descriptions,
        "orientations": orientation_table.as_dict(),
        "provenance": run_provenance(effective_config(config)),
    }

    write_score_file(output_path, rows, metadata, indent=None)
    free(bundle.unlearned, bundle.finetuned, bundle.base, bundle.shadow)

    print(f"\nWrote {len(rows)} rows to {output_path}")
    return output_path


def _fuma_discrepancies(config: Mapping[str, Any], collector: ScoreCollector) -> List[str]:
    """The record of how this FUMA implementation departs from the published one."""
    notes = [
        "the exact published FUMA formula was not verifiable from this "
        "environment; the convention used here is stated explicitly "
        f"(norm={config['scores']['fuma']['norm']}, "
        f"weighting={config['scores']['fuma']['weighting']}) and is configurable "
        "-- see docs/FUMA.md",
        "LoRA membership is decided by the substring 'lora' in the parameter "
        "name, matching PEFT's own naming",
    ]
    if collector.fuma_source == "shadow_bf16":
        notes.extend(
            [
                "both the numerator and the denominator are taken from an "
                "unquantised bf16 copy of the unlearned model, not from the 4-bit "
                "model that was actually unlearned: bitsandbytes does not "
                "backpropagate into 4-bit weights, so the denominator does not "
                "exist on the scored model",
                "the backbone is unfrozen purely to obtain gradients; it was "
                "frozen during training, so these gradients describe a direction "
                "the optimiser never took",
                "the non-LoRA group includes the embedding and output projection, "
                "which dominate the parameter-weighted denominator",
            ]
        )
    if collector.fuma_unfroze_backbone:
        notes.append(
            "the frozen backbone of the scored model was unfrozen to obtain the "
            "denominator; it was frozen during training, so these gradients "
            "describe a direction the optimiser never took"
        )
    return notes


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect per-sample deletion-set inference scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  $PY -m deletion_signals.collect \\\n"
            "      --config llama_lora.yaml --seed 42 --method ga_gd \\\n"
            "      --unlearned-checkpoint ds_results/seed42/ga_gd/unlearned_model \\\n"
            "      --finetuned-checkpoint ds_checkpoints/finetuned_seed42 \\\n"
            "      --manifest ds_results/seed42/splits.json \\\n"
            "      --output ds_results/seed42/ga_gd/per_sample_scores.json\n"
        ),
    )
    parser.add_argument("--config", default=None, help="YAML config (name or path)")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="Override a config key, e.g. --set scores.sif.enabled=true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--method", required=True, help="Unlearning method label for the rows")
    parser.add_argument("--unlearned-checkpoint", required=True)
    parser.add_argument("--finetuned-checkpoint", required=True)
    parser.add_argument("--manifest", default=None, help="Split manifest JSON")
    parser.add_argument("--output", required=True, help="Destination per_sample_scores.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap samples per population (smoke runs)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    if args.seed is not None:
        config["seed"] = args.seed
    if args.max_samples is not None:
        config["scoring"]["max_samples_per_population"] = args.max_samples
    config = apply_cli_overrides(config, args.overrides)

    collect_scores(
        config=config,
        seed=int(config["seed"]),
        method=args.method,
        unlearned_checkpoint=args.unlearned_checkpoint,
        finetuned_checkpoint=args.finetuned_checkpoint,
        output_path=args.output,
        manifest_path=args.manifest,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
