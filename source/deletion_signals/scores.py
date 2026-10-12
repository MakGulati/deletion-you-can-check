"""Per-sample score definitions.

Deliberately torch-free: every function here takes floats and returns floats, so
the definitions can be unit-tested in milliseconds and recomputed offline from a
saved table without loading a model.

Score inventory
---------------
============================  ============================================
name                          definition
============================  ============================================
``loss_unlearned``            mean token NLL under the unlearned model
``loss_base``                 mean token NLL under the difficulty reference
``loss_ratio``                loss_unlearned / max(loss_base, eps)
``loss_delta``                loss_unlearned - loss_base
``loss_finetuned``            mean token NLL under the pre-unlearning model
``loss_ratio_vs_finetuned``   loss_unlearned / max(loss_finetuned, eps)
``loss_delta_vs_finetuned``   loss_unlearned - loss_finetuned
``grad_norm_sq``              ||∇L_unlearned(z)||²
``grad_norm_sq_ref``          ||∇L_ref(z)||² on the gradient reference model
``grad_ratio``                grad_norm_sq / max(grad_norm_sq_ref, eps)
``log_grad_ratio``            log(grad_norm_sq + eps) - log(grad_norm_sq_ref + eps)
``fuma_lora_ratio``           mean |grad| on LoRA / mean |grad| off LoRA
``raw_cosine``                cos(∇L_unlearned(z), ĝ), signed
``alignment_cos_sq``          cos²(∇L_unlearned(z), ĝ)
``sif``                       ∇Lᵀ (F + λI)⁻¹ ∇L  (ablation; see sif.py)
``combined_log_score``        β·log(sif) + log(alignment_cos_sq)
============================  ============================================

Two loss references, and why they are separate families
-------------------------------------------------------
``loss_base`` and ``loss_finetuned`` are both computed, and the derived scores
built on them are *not* interchangeable:

* ``loss_ratio`` / ``loss_delta`` divide out intrinsic sample difficulty against
  a model that never saw any of this data. What is left is **membership**: a
  fitted sample is easier for the target than for the base model. Nothing in
  this pair can see the unlearning direction.
* ``loss_ratio_vs_finetuned`` / ``loss_delta_vs_finetuned`` divide out the
  fine-tuned model's own fit, so what is left is the change caused
  **specifically by unlearning** -- a forget sample rose because of ascent, a
  retain sample fell because of descent, an unseen sample barely moved.

The same two numeric helpers compute both pairs; only the reference differs.

Orientations for each of these -- which direction means "more likely forget" --
are declared ahead of time in ``configs/orientations.yaml`` and are NOT chosen
after seeing results. See docs/ORIENTATIONS.md.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional, Tuple

# Floor for denominators. Chosen well below any loss or squared gradient norm
# a language model produces (both are O(1) to O(1e4) here) so it never
# perturbs a real value, while still bounding the ratio when a denominator
# genuinely underflows to zero.
DEFAULT_EPS = 1e-12


class FumaNorm(str, Enum):
    """How a group of gradient tensors is reduced to one magnitude.

    ``L1_MEAN_ABS`` mean absolute gradient entry, Σ|g| / Σ numel. The default.
    ``L2``          root mean square entry, sqrt(Σ g² / Σ numel).
    ``L2_SQ``       mean squared entry, Σ g² / Σ numel. Matches the convention
                    ``grad_norm_sq`` uses, so FUMA and the magnitude baselines
                    sit on the same scale.
    """

    L1_MEAN_ABS = "l1_mean_abs"
    L2 = "l2"
    L2_SQ = "l2_sq"


class FumaWeighting(str, Enum):
    """Whether "mean magnitude" averages over parameters or over tensors.

    ``PARAMETER`` divides by the total element count of the group. A 2048×2048
                  projection then outweighs a 16×2048 LoRA_A by its size, which
                  is what "mean gradient magnitude over the parameters" says.
    ``TENSOR``    averages each tensor's own per-entry mean, then averages those
                  across tensors, so every tensor is one vote regardless of
                  size.
    """

    PARAMETER = "parameter"
    TENSOR = "tensor"


def loss_ratio(
    loss_unlearned: float, loss_base: Optional[float], eps: float = DEFAULT_EPS
) -> Optional[float]:
    """Difficulty-calibrated loss: L_unlearned(z) / max(L_base(z), eps).

    This is a *membership / difficulty* baseline, not a deletion-set signal.
    It asks "is this sample harder for the target than for a model that never
    saw it", which separates members from non-members. It has no access to the
    unlearning direction, so it cannot by construction tell a forget member from
    a retain member except through whatever collateral damage unlearning did to
    the forget set's loss. That is exactly why it belongs in the comparison.

    Deliberately not called LiRA: LiRA fits per-sample Gaussians over many shadow
    models and does a likelihood-ratio test. This is one reference model and a
    ratio.

    The same function computes ``loss_ratio_vs_finetuned`` when the reference is
    the pre-unlearning fine-tuned model instead of the pretrained base. That is a
    different quantity with a different meaning -- the change unlearning caused,
    not membership -- which is why it is recorded under its own column name
    rather than folded into this one.
    """
    if loss_base is None or not math.isfinite(loss_base):
        return None
    if not math.isfinite(loss_unlearned):
        return None
    return loss_unlearned / max(loss_base, eps)


def loss_delta(loss_unlearned: float, loss_base: Optional[float]) -> Optional[float]:
    """L_unlearned(z) - L_base(z). Same calibration idea, additive."""
    if loss_base is None or not math.isfinite(loss_base):
        return None
    if not math.isfinite(loss_unlearned):
        return None
    return loss_unlearned - loss_base


def grad_ratio(
    grad_norm_sq: float, grad_norm_sq_ref: Optional[float], eps: float = DEFAULT_EPS
) -> Optional[float]:
    """||∇L_unlearned(z)||² / max(||∇L_ref(z)||², eps)."""
    if grad_norm_sq_ref is None or not math.isfinite(grad_norm_sq_ref):
        return None
    if not math.isfinite(grad_norm_sq):
        return None
    return grad_norm_sq / max(grad_norm_sq_ref, eps)


def log_grad_ratio(
    grad_norm_sq: float, grad_norm_sq_ref: Optional[float], eps: float = DEFAULT_EPS
) -> Optional[float]:
    """log(||∇L_unl||² + eps) - log(||∇L_ref||² + eps).

    Not simply ``log(grad_ratio)``: the additive eps keeps the score finite when
    *both* norms underflow, where the ratio form would return 1.0 and imply "no
    change" rather than "no signal". The two agree to machine precision whenever
    both norms are well above eps, which is the ordinary case.
    """
    if grad_norm_sq_ref is None or not math.isfinite(grad_norm_sq_ref):
        return None
    if not math.isfinite(grad_norm_sq):
        return None
    return math.log(max(grad_norm_sq, 0.0) + eps) - math.log(
        max(grad_norm_sq_ref, 0.0) + eps
    )


def cos_sq(raw_cosine: Optional[float]) -> Optional[float]:
    """cos², clamped into [0, 1].

    Sign-invariant, so the ĝ sign convention (θ_unlearned − θ_finetuned rather
    than its negation) does not affect this score. ``raw_cosine`` is recorded
    separately for anyone who wants the signed quantity.
    """
    if raw_cosine is None or not math.isfinite(raw_cosine):
        return None
    return min(1.0, max(0.0, raw_cosine * raw_cosine))


def combined_log_score(
    sif: Optional[float], alignment_cos_sq: Optional[float], beta: float = 1.0
) -> Optional[float]:
    """β·log(SIF) + log(cos²) -- the existing combined score, ranked in log space.

    The direct product underflows: SIF is O(1e1..1e4) while cos² is O(1e-4) or
    smaller, and float32 loses the low-order bits that the rank statistic
    depends on. ``-inf`` is the correct ranking position for a sample with zero
    alignment: below every sample that has some.
    """
    if alignment_cos_sq is None:
        return None
    if alignment_cos_sq <= 0.0:
        return float("-inf")
    if beta == 0.0:
        return math.log(alignment_cos_sq)
    if sif is None:
        return None
    if sif <= 0.0:
        # Only reachable under the legacy Hessian estimator, where a negative
        # quadratic form is the genuine (and genuinely ill-defined) value.
        return float("-inf")
    return beta * math.log(sif) + math.log(alignment_cos_sq)


def _group_magnitude(
    abs_sum: float,
    sq_sum: float,
    numel: int,
    tensor_abs_mean_sum: float,
    n_tensors: int,
    norm: FumaNorm,
    weighting: FumaWeighting,
) -> Optional[float]:
    """Reduce one parameter group's accumulators to a single magnitude."""
    if weighting is FumaWeighting.TENSOR:
        if n_tensors == 0:
            return None
        if norm is not FumaNorm.L1_MEAN_ABS:
            # Tensor weighting only has a defined meaning for the mean-|g|
            # reduction with the accumulators we keep; anything else would need
            # per-tensor squared sums, which are not worth carrying per sample.
            raise ValueError(
                "FumaWeighting.TENSOR is only supported with "
                "FumaNorm.L1_MEAN_ABS; got norm=" + norm.value
            )
        return tensor_abs_mean_sum / n_tensors

    if numel == 0:
        return None
    if norm is FumaNorm.L1_MEAN_ABS:
        return abs_sum / numel
    if norm is FumaNorm.L2:
        return math.sqrt(max(sq_sum, 0.0) / numel)
    return sq_sum / numel  # L2_SQ


def fuma_lora_ratio(
    abs_sum_lora: float,
    abs_sum_non_lora: float,
    sq_sum_lora: float,
    sq_sum_non_lora: float,
    numel_lora: int,
    numel_non_lora: int,
    tensor_abs_mean_sum_lora: float = 0.0,
    tensor_abs_mean_sum_non_lora: float = 0.0,
    n_tensors_lora: int = 0,
    n_tensors_non_lora: int = 0,
    norm: FumaNorm | str = FumaNorm.L1_MEAN_ABS,
    weighting: FumaWeighting | str = FumaWeighting.PARAMETER,
    eps: float = DEFAULT_EPS,
) -> Tuple[Optional[float], Optional[str]]:
    """FUMA-style LoRA gradient ratio, or an explicit not-applicable reason.

        ratio(z) = magnitude(∇L(z) restricted to LoRA parameters)
                 / max(magnitude(∇L(z) restricted to non-LoRA parameters), eps)

    Returns ``(value, na_reason)``. Exactly one of the two is ``None``. A
    not-applicable case is never reported as 0.0 or 1.0 -- both are plausible
    ratios, and a reader cannot tell a real measurement from a placeholder.

    Not-applicable cases, and why they arise:

    * **No LoRA parameters in scope.** A fully fine-tuned checkpoint has no
      adapters, so the numerator is undefined. A defensible
      updated-vs-non-updated analogue exists, but it is a different quantity
      and is exposed under a different name -- see
      :func:`updated_parameter_ratio` -- rather than being labelled FUMA.
    * **No non-LoRA gradients.** Under QLoRA the backbone is frozen and 4-bit
      quantised; bitsandbytes does not backpropagate into 4-bit weights, so the
      denominator does not exist in that parameter space. The collector's bf16
      shadow model supplies it instead; if the shadow model is disabled or does
      not fit in memory, this reason is what gets recorded.

    See docs/FUMA.md for the exact conventions and their known deviations from
    the published setting.
    """
    norm = FumaNorm(norm)
    weighting = FumaWeighting(weighting)

    numerator = _group_magnitude(
        abs_sum_lora,
        sq_sum_lora,
        numel_lora,
        tensor_abs_mean_sum_lora,
        n_tensors_lora,
        norm,
        weighting,
    )
    if numerator is None:
        return None, (
            "no LoRA parameters in the scored parameter space (full fine-tuning "
            "or a non-LoRA checkpoint); FUMA's LoRA/non-LoRA split is undefined"
        )

    denominator = _group_magnitude(
        abs_sum_non_lora,
        sq_sum_non_lora,
        numel_non_lora,
        tensor_abs_mean_sum_non_lora,
        n_tensors_non_lora,
        norm,
        weighting,
    )
    if denominator is None:
        return None, (
            "no non-LoRA parameter received a gradient: under QLoRA the backbone "
            "is frozen and 4-bit quantised, and bitsandbytes does not "
            "backpropagate into 4-bit weights. Enable the bf16 shadow model "
            "(scores.fuma.denominator_source=shadow_bf16) to obtain it."
        )

    if not (math.isfinite(numerator) and math.isfinite(denominator)):
        return None, "numerator or denominator is not finite"

    return numerator / max(denominator, eps), None


def updated_parameter_ratio(
    magnitude_updated: Optional[float],
    magnitude_frozen: Optional[float],
    eps: float = DEFAULT_EPS,
) -> Tuple[Optional[float], Optional[str]]:
    """Full-fine-tuning analogue of the FUMA ratio, under a different name.

    In a full fine-tune every parameter is updated, so there is no
    updated-vs-frozen partition and this too is not applicable unless the caller
    supplies a partition of its own (a named layer subset via
    ``ParamScope.NAMED_SUBSET``). It exists so that a full-fine-tuning run can
    report *something* comparable without that number being labelled FUMA --
    the published method is defined over LoRA adapters, and calling a different
    partition by its name would misreport the comparison.
    """
    if magnitude_updated is None or magnitude_frozen is None:
        return None, (
            "no updated/non-updated parameter partition is defined for this run; "
            "supply one via param_scope=named_subset to enable this ratio"
        )
    if not (math.isfinite(magnitude_updated) and math.isfinite(magnitude_frozen)):
        return None, "numerator or denominator is not finite"
    return magnitude_updated / max(magnitude_frozen, eps), None
