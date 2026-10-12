"""Self-influence, kept as an ablation.

    SIF(z) = ∇L(z)ᵀ (C + λI)⁻¹ ∇L(z)

with two estimators for the curvature C:

``gradient``  C = 0, so SIF reduces to ``||∇L(z)||² / λ`` -- a rescaled gradient
              norm, and therefore rank-identical to ``grad_norm_sq``. Cheap, and
              the honest default: it makes explicit that the "curvature-weighted"
              score is a gradient norm whenever the curvature term is inert.

``fisher``    C = (1/n) GᵀG, the empirical Fisher over n held-out retain
              gradients stacked in G ∈ R^{n×d}. Positive semi-definite by
              construction, so (C + λI) ≻ 0 and SIF > 0 always. Solved exactly by
              the Woodbury identity:

                  SIF(z) = (1/λ)[ ||g||² − (Gg)ᵀ (nλI + GGᵀ)⁻¹ (Gg) ]

              The n×n matrix is factorised once, so a query costs one backward
              pass plus an n-dimensional solve.

The Hessian estimator from the original pipeline is deliberately not
reimplemented: away from a minimum H is indefinite, (H + λI) is not positive
definite, and the resulting quadratic form is negative for most samples, which
inverts the ranking wherever it is multiplied into the combined score. It was
retained there as a recorded negative result; reproducing it here would add a
known-broken estimator to a comparison that has no use for one.

``fisher_explained`` -- the share of ||∇L(z)||² that the rank-n Fisher actually
resolves -- is reported alongside every value. rank(C) ≤ n and n is tiny against
d, so if that fraction is near zero then (C + λI)⁻¹ → I/λ and every "SIF" in the
run is a gradient norm wearing a hat.

CURVATURE SAMPLES MUST BE DISJOINT FROM SCORED SAMPLES. A sample whose gradient
is a row of G lies in the row space of C and is damped hardest, while a sample
outside that span approaches the 1/λ ceiling. Scoring the curvature samples
manufactures a retain-versus-unseen separation that is pure leakage.
:meth:`FisherSelfInfluence.assert_disjoint` enforces it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set

import torch


@dataclass
class SifValue:
    """One self-influence measurement."""

    value: float
    fisher_explained: float = float("nan")


def gradient_self_influence(grad_norm_sq: float, damping: float) -> SifValue:
    """The C = 0 limit: ``||∇L(z)||² / λ``.

    Reported with the damping divided out so the number is on the same scale as
    the Fisher estimator's. It is a strictly monotone function of
    ``grad_norm_sq``, so its AUC equals that of the raw gradient norm on every
    task -- which is the point of including it.
    """
    if damping <= 0:
        raise ValueError(f"damping must be positive, got {damping}")
    if not math.isfinite(grad_norm_sq):
        return SifValue(float("nan"))
    return SifValue(grad_norm_sq / damping)


class FisherSelfInfluence:
    """Empirical-Fisher self-influence, solved exactly via Woodbury.

    Holds G (n × d) and the Cholesky factor of (nλI + GGᵀ). G is the only large
    object in the evaluation: at n=200 and d≈3.4M it is ~2.7 GB in float32,
    which competes with the model for VRAM, so it defaults to the CPU.
    """

    def __init__(
        self,
        damping: float = 1e-3,
        rank: int = 200,
        device: str = "cpu",
    ):
        if damping <= 0:
            raise ValueError(
                f"Fisher damping must be positive to keep (F + lambda I) "
                f"invertible, got {damping}"
            )
        self.damping = float(damping)
        self.rank = int(rank)
        self.device = torch.device(device)

        self._G: Optional[torch.Tensor] = None
        self._chol: Optional[torch.Tensor] = None
        self._fingerprints: Set[str] = set()

    @property
    def is_built(self) -> bool:
        return self._G is not None and self._chol is not None

    def build(
        self,
        gradients: Iterable[torch.Tensor],
        fingerprints: Optional[Sequence[str]] = None,
    ) -> None:
        """Form G from held-out gradients and factorise (nλI + GGᵀ).

        Args:
            gradients: Flat per-sample gradients from the curvature holdout.
            fingerprints: Their ``text_sha1`` values, so
                :meth:`assert_disjoint` can catch overlap with the scored set.
        """
        rows: List[torch.Tensor] = []
        for gradient in gradients:
            rows.append(gradient.detach().reshape(-1).to(torch.float32).to(self.device))
            if len(rows) >= self.rank:
                break

        if not rows:
            raise ValueError("Cannot build the Fisher estimate from zero gradients.")

        G = torch.stack(rows)
        n = G.shape[0]

        # M = nλI + GGᵀ, in float64: the n × n solve is the only step where
        # conditioning is genuinely at risk, and it is cheap enough to do in
        # double regardless of n.
        M = (G @ G.T).double()
        M.diagonal().add_(n * self.damping)
        self._chol = torch.linalg.cholesky(M)
        self._G = G
        self._fingerprints = set(fingerprints or ())

    def assert_disjoint(self, fingerprints: Iterable[str]) -> None:
        """Fail if any sample to be scored also went into the curvature estimate.

        Overlap raises nothing downstream -- it silently inflates the
        retain-versus-unseen AUC, the one number the three-way framing rests on.
        It has to fail here or it will not fail at all.
        """
        if not self._fingerprints:
            return
        overlap = self._fingerprints.intersection(f for f in fingerprints if f)
        if overlap:
            raise ValueError(
                f"{len(overlap)} sample(s) to be scored were also used to build "
                f"the Fisher estimate. Those samples lie in the row space of G "
                f"and are damped hardest, which fabricates a retain-vs-unseen "
                f"gap. Hold out a disjoint curvature slice."
            )

    def score(self, gradient: torch.Tensor, grad_norm_sq: float) -> SifValue:
        """SIF(z) and the explained fraction, from a flat gradient."""
        if not self.is_built:
            raise RuntimeError(
                "Fisher estimate not built; call build() with held-out gradients."
            )
        if not math.isfinite(grad_norm_sq):
            return SifValue(float("nan"), float("nan"))

        g = gradient.detach().reshape(-1).to(self.device, dtype=torch.float32)
        Gg = (self._G @ g).double().unsqueeze(1)
        y = torch.cholesky_solve(Gg, self._chol)
        quad = float((Gg * y).sum())

        value = (grad_norm_sq - quad) / self.damping
        explained = quad / grad_norm_sq if grad_norm_sq > 0 else float("nan")

        # Mathematically SIF > 0. A small negative here is float error on a
        # near-zero gradient, so clamping is safe -- which it explicitly is NOT
        # for a Hessian-based estimator, where negative values are the real
        # quantity and hiding them would hide the negative result.
        return SifValue(max(value, 0.0), explained)

    def memory_bytes(self) -> int:
        return 0 if self._G is None else self._G.numel() * self._G.element_size()
