"""Parameter selection and layerwise float64 gradient statistics.

Every quantity the evaluation compares -- alignment, gradient magnitude, the
FUMA LoRA ratio -- is a reduction over the same per-sample gradient. This module
computes all of them in a single pass over the named parameters, accumulating in
float64, and never materialising a concatenated d-dimensional vector.

Why float64 and why layerwise
-----------------------------
``cos(g, ghat)`` is a ratio of sums over millions of terms. The model runs in
bfloat16 and gradients arrive in bfloat16 or float32; accumulating a
3.4M-term dot product in that precision loses enough low-order bits to move the
cosine, and the cosine is what the headline AUC ranks by. Casting each parameter
tensor to float64 and summing per tensor keeps the running totals exact to
~15 digits while the peak extra memory is one tensor, not one model.

``concatenated_gradient_stats`` computes the same numbers the naive way (build
the flat vector, then reduce). It exists so the layerwise path can be tested
against it -- see ``tests/test_gradients.py``.
"""

from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

# PEFT names adapter parameters like
#   base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
# so a lowercase substring test on "lora" identifies them. This is the same
# convention the rest of the unlearning literature (and PEFT's own
# ``get_peft_model_state_dict``) relies on. Anything else -- embeddings, the
# frozen backbone projections, layer norms -- counts as non-LoRA.
LORA_NAME_MARKER = "lora"


def is_lora_param(name: str) -> bool:
    """True if ``name`` denotes a LoRA adapter parameter."""
    return LORA_NAME_MARKER in name.lower()


class ParamScope(str, Enum):
    """Which parameters participate in the gradient statistics.

    ``TRAINABLE``     every parameter with ``requires_grad`` (the default).
    ``LORA``          adapter parameters only; the space ĝ = θ_unl − θ_ft lives
                      in for a QLoRA run.
    ``FULL``          every parameter, trainable or not. Only meaningful when
                      something has actually populated ``.grad`` for them --
                      i.e. the bf16 shadow model, not the 4-bit backbone.
    ``NAMED_SUBSET``  parameters whose names match one of ``name_patterns``
                      (``fnmatch`` globs), for layer-subset studies.
    """

    TRAINABLE = "trainable"
    LORA = "lora"
    FULL = "full"
    NAMED_SUBSET = "named_subset"


def select_param_names(
    model: torch.nn.Module,
    scope: ParamScope | str = ParamScope.TRAINABLE,
    name_patterns: Optional[Sequence[str]] = None,
) -> List[str]:
    """Names of the parameters in ``scope``, in ``named_parameters()`` order.

    Order matters: it is the order the concatenated reference implementation
    uses, and two models are only comparable if they enumerate the same names.

    Raises:
        ValueError: if the selection is empty, which otherwise surfaces much
            later as an all-zero gradient and a cosine of 0 for every sample.
    """
    scope = ParamScope(scope)
    if scope is ParamScope.NAMED_SUBSET and not name_patterns:
        raise ValueError("ParamScope.NAMED_SUBSET requires non-empty name_patterns")

    names: List[str] = []
    for name, param in model.named_parameters():
        if scope is ParamScope.TRAINABLE and not param.requires_grad:
            continue
        if scope is ParamScope.LORA and not is_lora_param(name):
            continue
        if scope is ParamScope.NAMED_SUBSET and not any(
            fnmatch.fnmatch(name, pattern) for pattern in name_patterns or ()
        ):
            continue
        names.append(name)

    if not names:
        raise ValueError(
            f"No parameters matched scope={scope.value}"
            + (f" patterns={list(name_patterns)}" if name_patterns else "")
            + ". For a QLoRA checkpoint the model must expose trainable "
            "parameters whose names contain 'lora' (load it with "
            "is_trainable=True); for a fully fine-tuned checkpoint use "
            "scope='trainable' or 'full'."
        )
    return names


@dataclass
class GradientStats:
    """Reductions over one per-sample gradient, accumulated in float64.

    All sums are plain Python floats so the object is JSON-serialisable and
    carries no device or dtype baggage into the output rows.
    """

    # ||g||^2 and, when a direction was supplied, <g, ghat> and ||ghat||^2.
    norm_sq: float = 0.0
    dot_direction: float = 0.0
    direction_norm_sq: float = 0.0

    # Parameter-weighted accumulators, split by LoRA membership. These feed the
    # FUMA ratio; see scores.fuma_lora_ratio for how they combine.
    abs_sum_lora: float = 0.0
    abs_sum_non_lora: float = 0.0
    sq_sum_lora: float = 0.0
    sq_sum_non_lora: float = 0.0
    numel_lora: int = 0
    numel_non_lora: int = 0

    # Tensor-weighted accumulators: the sum over tensors of that tensor's own
    # per-parameter mean magnitude. Dividing by the tensor count gives a mean
    # that treats a 16x2048 LoRA_A and a 2048x2048 projection as equal votes,
    # which is the other reading of "mean gradient magnitude".
    tensor_abs_mean_sum_lora: float = 0.0
    tensor_abs_mean_sum_non_lora: float = 0.0
    n_tensors_lora: int = 0
    n_tensors_non_lora: int = 0

    # Validity bookkeeping. A parameter in scope whose .grad is None
    # contributes zeros (it did not participate in the loss) and is counted;
    # a tensor containing NaN/Inf is counted AND allowed to poison the sums, so
    # downstream scores come out non-finite and get flagged rather than
    # silently rounded into a plausible-looking number.
    n_params: int = 0
    n_missing_grad: int = 0
    n_nonfinite_tensors: int = 0

    @property
    def numel_total(self) -> int:
        return self.numel_lora + self.numel_non_lora

    @property
    def is_finite(self) -> bool:
        values = (
            self.norm_sq,
            self.dot_direction,
            self.direction_norm_sq,
            self.abs_sum_lora,
            self.abs_sum_non_lora,
        )
        return self.n_nonfinite_tensors == 0 and all(math.isfinite(v) for v in values)

    def raw_cosine(self, eps: float = 1e-30) -> float:
        """cos(g, ĝ), clamped into [-1, 1].

        Returns 0.0 when either vector has (numerically) zero norm: an
        undefined direction carries no alignment information, and 0.0 places
        such a sample at the neutral point of the cos² ranking rather than at
        an extreme. Callers should also consult ``zero_norm`` to distinguish
        "no alignment" from "no gradient at all".
        """
        denom_sq = self.norm_sq * self.direction_norm_sq
        if not math.isfinite(denom_sq) or denom_sq <= eps:
            return 0.0
        cosine = self.dot_direction / math.sqrt(denom_sq)
        return max(-1.0, min(1.0, cosine))

    def zero_norm(self, eps: float = 1e-30) -> bool:
        """True if the gradient or the direction is numerically zero."""
        return not (self.norm_sq > eps and self.direction_norm_sq > eps)


def accumulate_gradient_stats(
    named_grads: Iterable[Tuple[str, Optional[torch.Tensor]]],
    direction: Optional[Mapping[str, torch.Tensor]] = None,
    lora_predicate: Callable[[str], bool] = is_lora_param,
    numel_hint: Optional[Mapping[str, int]] = None,
) -> GradientStats:
    """Reduce a per-sample gradient to :class:`GradientStats`, layer by layer.

    Args:
        named_grads: ``(name, grad)`` pairs. ``grad`` may be ``None`` for a
            parameter that did not receive one; it then contributes zeros to
            every sum, and its element count still counts towards the FUMA
            denominators if ``numel_hint`` supplies it.
        direction: ``name -> ĝ`` slices, typically θ_unlearned − θ_finetuned.
            Names absent from this mapping contribute nothing to
            ``dot_direction`` or ``direction_norm_sq``; a name present here but
            not in ``named_grads`` is ignored, so a mismatch cannot silently
            inflate ``||ĝ||``.
        lora_predicate: classifies a parameter name as LoRA or not.
        numel_hint: element counts for parameters whose gradient is ``None``.
            Without it a missing gradient also removes those parameters from
            the FUMA denominator, which would inflate the ratio.

    Returns:
        GradientStats with every sum accumulated in float64.
    """
    stats = GradientStats()
    numel_hint = numel_hint or {}

    for name, grad in named_grads:
        stats.n_params += 1
        is_lora = lora_predicate(name)

        if grad is None:
            stats.n_missing_grad += 1
            numel = int(numel_hint.get(name, 0))
            if is_lora:
                stats.numel_lora += numel
                stats.n_tensors_lora += 1
            else:
                stats.numel_non_lora += numel
                stats.n_tensors_non_lora += 1
            # A missing gradient still contributes its direction component:
            # ĝ is a property of the two checkpoints, not of this sample.
            direction_slice = None if direction is None else direction.get(name)
            if direction_slice is not None:
                d64 = direction_slice.detach().reshape(-1).to(torch.float64)
                stats.direction_norm_sq += float(torch.dot(d64, d64))
            continue

        g64 = grad.detach().reshape(-1).to(torch.float64)
        numel = g64.numel()

        if not bool(torch.isfinite(g64).all()):
            stats.n_nonfinite_tensors += 1

        sq = float(torch.dot(g64, g64))
        abs_sum = float(g64.abs().sum())
        stats.norm_sq += sq

        if is_lora:
            stats.numel_lora += numel
            stats.n_tensors_lora += 1
            stats.sq_sum_lora += sq
            stats.abs_sum_lora += abs_sum
            stats.tensor_abs_mean_sum_lora += abs_sum / numel if numel else 0.0
        else:
            stats.numel_non_lora += numel
            stats.n_tensors_non_lora += 1
            stats.sq_sum_non_lora += sq
            stats.abs_sum_non_lora += abs_sum
            stats.tensor_abs_mean_sum_non_lora += abs_sum / numel if numel else 0.0

        direction_slice = None if direction is None else direction.get(name)
        if direction_slice is not None:
            d64 = direction_slice.detach().reshape(-1).to(torch.float64)
            if d64.numel() != numel:
                raise ValueError(
                    f"Shape mismatch for parameter {name!r}: gradient has "
                    f"{numel} entries, unlearning direction has {d64.numel()}. "
                    f"The unlearned and fine-tuned checkpoints must expose the "
                    f"same parameter space."
                )
            stats.dot_direction += float(torch.dot(g64, d64))
            stats.direction_norm_sq += float(torch.dot(d64, d64))

    return stats


def concatenated_gradient_stats(
    named_grads: Sequence[Tuple[str, Optional[torch.Tensor]]],
    direction: Optional[Mapping[str, torch.Tensor]] = None,
) -> Tuple[float, float, float]:
    """Reference implementation: build the flat vectors, then reduce.

    Returns ``(norm_sq, dot_direction, direction_norm_sq)``. Used only by the
    tests, to pin ``accumulate_gradient_stats`` against the obvious-but-costly
    formulation it replaces.
    """
    grad_parts: List[torch.Tensor] = []
    dir_parts: List[torch.Tensor] = []

    for name, grad in named_grads:
        direction_slice = None if direction is None else direction.get(name)
        if grad is None:
            if direction_slice is None:
                continue
            flat_dir = direction_slice.detach().reshape(-1).to(torch.float64)
            grad_parts.append(torch.zeros_like(flat_dir))
            dir_parts.append(flat_dir)
            continue

        flat_grad = grad.detach().reshape(-1).to(torch.float64)
        grad_parts.append(flat_grad)
        dir_parts.append(
            torch.zeros_like(flat_grad)
            if direction_slice is None
            else direction_slice.detach().reshape(-1).to(torch.float64)
        )

    if not grad_parts:
        return 0.0, 0.0, 0.0

    g = torch.cat(grad_parts)
    d = torch.cat(dir_parts)
    return float(torch.dot(g, g)), float(torch.dot(g, d)), float(torch.dot(d, d))


def collect_named_grads(
    model: torch.nn.Module, names: Sequence[str]
) -> List[Tuple[str, Optional[torch.Tensor]]]:
    """``(name, .grad)`` for each requested parameter, in ``names`` order."""
    lookup = dict(model.named_parameters())
    missing = [n for n in names if n not in lookup]
    if missing:
        raise KeyError(
            f"{len(missing)} requested parameter(s) are absent from the model, "
            f"first few: {missing[:3]}"
        )
    return [(n, lookup[n].grad) for n in names]


def parameter_numels(
    model: torch.nn.Module, names: Sequence[str]
) -> Dict[str, int]:
    """Element count per requested parameter, for the ``numel_hint`` argument."""
    lookup = dict(model.named_parameters())
    return {n: lookup[n].numel() for n in names if n in lookup}


def parameter_difference(
    unlearned: torch.nn.Module,
    finetuned: torch.nn.Module,
    names: Sequence[str],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """ĝ = θ_unlearned − θ_finetuned, as a per-parameter mapping.

    Kept as a mapping rather than a flat vector so the layerwise accumulation
    can consume it directly, and so a name mismatch between the two checkpoints
    fails here with a name attached instead of as a length mismatch later.

    Sign convention: this is the *displacement produced by unlearning*, so it
    points from the fine-tuned model towards the unlearned one. cos² is
    invariant to the sign, so the convention affects only ``raw_cosine``; it is
    recorded in the output metadata and documented in docs/ORIENTATIONS.md.
    """
    unlearned_params = dict(unlearned.named_parameters())
    finetuned_params = dict(finetuned.named_parameters())

    direction: Dict[str, torch.Tensor] = {}
    for name in names:
        if name not in unlearned_params:
            raise KeyError(f"{name!r} is absent from the unlearned model")
        if name not in finetuned_params:
            raise KeyError(
                f"{name!r} is absent from the fine-tuned model. Both checkpoints "
                f"must expose the same parameter names for PARAM_DIFF."
            )
        theta_u = unlearned_params[name].detach()
        theta_f = finetuned_params[name].detach().to(theta_u.device)
        if theta_u.shape != theta_f.shape:
            raise ValueError(
                f"Shape mismatch for {name!r}: unlearned {tuple(theta_u.shape)} "
                f"vs fine-tuned {tuple(theta_f.shape)}"
            )
        delta = (theta_u.to(dtype) - theta_f.to(dtype)).reshape(-1)
        direction[name] = delta.to(device) if device is not None else delta

    return direction


def direction_norm(direction: Mapping[str, torch.Tensor]) -> float:
    """||ĝ||, accumulated in float64 across the mapping."""
    total = 0.0
    for tensor in direction.values():
        t64 = tensor.detach().reshape(-1).to(torch.float64)
        total += float(torch.dot(t64, t64))
    return math.sqrt(total)


@dataclass
class FlatGradientBuffer:
    """A concatenated float32 gradient, for the self-influence ablation only.

    The empirical-Fisher SIF needs the actual vector (it forms G ∈ R^{n×d}), so
    that one score cannot use the layerwise path. Every other score can, which
    is why this is opt-in rather than the default.
    """

    names: Tuple[str, ...]
    sizes: Tuple[int, ...] = field(default_factory=tuple)

    @classmethod
    def for_model(cls, model: torch.nn.Module, names: Sequence[str]) -> "FlatGradientBuffer":
        lookup = dict(model.named_parameters())
        return cls(tuple(names), tuple(lookup[n].numel() for n in names))

    @property
    def dim(self) -> int:
        return sum(self.sizes)

    def flatten(
        self,
        named_grads: Sequence[Tuple[str, Optional[torch.Tensor]]],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Concatenate the gradients in ``self.names`` order, as float32."""
        by_name = dict(named_grads)
        parts: List[torch.Tensor] = []
        for name, size in zip(self.names, self.sizes):
            grad = by_name.get(name)
            if grad is None:
                parts.append(torch.zeros(size, dtype=torch.float32, device=device))
            else:
                parts.append(grad.detach().reshape(-1).to(torch.float32).to(device))
        return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.float32)
