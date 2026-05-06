from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class PowerSpec:
    iters: int
    tol: float


def topk_eigs_power(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    d: int,
    k: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    spec: PowerSpec,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Deflated power iteration for symmetric PSD operators.
    Returns (k,) eigenvalues (descending).
    """
    evals = []
    vecs = []

    def orthogonalize(v: torch.Tensor) -> torch.Tensor:
        for u in vecs:
            v = v - torch.dot(v, u) * u
        return v

    for _ in range(k):
        v = torch.randn((d,), device=device, dtype=dtype, generator=generator)
        v = orthogonalize(v)
        v = v / (torch.norm(v) + 1e-12)

        last_lambda = None
        for _it in range(spec.iters):
            w = matvec(v)
            w = orthogonalize(w)
            nw = torch.norm(w)
            if nw.item() == 0:
                break
            v = w / nw
            lam = torch.dot(v, matvec(v)).item()
            if last_lambda is not None and abs(lam - last_lambda) <= spec.tol * max(1.0, abs(last_lambda)):
                last_lambda = lam
                break
            last_lambda = lam

        evals.append(float(last_lambda if last_lambda is not None else 0.0))
        vecs.append(v.detach())

    return torch.tensor(evals, device=device, dtype=dtype)

