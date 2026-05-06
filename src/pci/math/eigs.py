from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EigsSpec:
    top_k: int
    method: str  # "exact"|"power"
    power_iters: int = 80
    power_tol: float = 1e-6


def topk_eigs_exact(cov: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    cov: (..., d, d) symmetric PSD
    returns eigvals: (..., top_k) sorted desc
    """
    evals = torch.linalg.eigvalsh(cov)  # (..., d) asc
    evals = torch.flip(evals, dims=[-1])
    return evals[..., :top_k]


def effective_dimension_participation_ratio(eigs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    eigs: (..., k) nonnegative
    d_eff = (sum λ)^2 / sum λ^2
    """
    s1 = torch.sum(eigs, dim=-1)
    s2 = torch.sum(eigs * eigs, dim=-1)
    return (s1 * s1) / (s2 + eps)

