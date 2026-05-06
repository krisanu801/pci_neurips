from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class GammaGrid:
    gamma_min: float
    gamma_max: float
    num: int

    def values(self) -> np.ndarray:
        gmin = float(self.gamma_min)
        gmax = float(self.gamma_max)
        n = int(self.num)
        return np.logspace(np.log10(gmin), np.log10(gmax), n)


def corrupt_standardized(x0: torch.Tensor, gamma: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
    """
    Standardized channel: Y = sqrt(gamma) * x0 + eps, eps ~ N(0, I).
    x0: (N, d)
    gamma: (G,) or scalar tensor
    returns: Y with shape (G, N, d) if gamma is (G,), else (N, d)
    """
    if gamma.ndim == 0:
        eps = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=generator)
        return torch.sqrt(gamma) * x0 + eps
    g = gamma[:, None, None]
    eps = torch.randn((gamma.shape[0],) + tuple(x0.shape), device=x0.device, dtype=x0.dtype, generator=generator)
    return torch.sqrt(g) * x0[None, :, :] + eps
