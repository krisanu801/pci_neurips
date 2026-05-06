from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AdapterBatch:
    y: torch.Tensor  # (N,d)
    gamma: float


class ModelAdapter(ABC):
    """
    Minimal interface needed to extract posterior-covariance spectra.

    The experiments treat each model family as a black box that can provide:
    - the implied denoiser D(gamma, y) ≈ E[x0 | y]
    - Jacobian-vector products (J_D v) for v in R^d, used in power iteration / Lanczos
    """

    model_family: str

    @abstractmethod
    def to(self, device: torch.device, dtype: torch.dtype) -> "ModelAdapter":
        raise NotImplementedError

    @abstractmethod
    def denoiser(self, batch: AdapterBatch) -> torch.Tensor:
        """Return D(gamma, y) with shape (N,d)."""
        raise NotImplementedError

    @abstractmethod
    def jvp_denoiser(self, batch: AdapterBatch, v: torch.Tensor) -> torch.Tensor:
        """
        Compute (J_D)v where J_D = ∂D/∂y at (gamma, y).
        v: (N,d) or (d,) broadcastable to (N,d)
        returns: (N,d)
        """
        raise NotImplementedError

