from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from pci.models.adapters.base import AdapterBatch, ModelAdapter


@dataclass
class OracleGaussianAdapter(ModelAdapter):
    """
    A fully analytic adapter for Exp.2 scaffolding tests:
    prior x0 ~ N(0, Sigma0), with diagonal Sigma0.

    This is NOT a pretrained image model; it's used to verify the Exp.2 pipeline
    (CSV schemas, eig extraction, pairwise comparisons) end-to-end.
    """

    model_family: str
    sigma0_diag: torch.Tensor  # (d,)

    def to(self, device: torch.device, dtype: torch.dtype) -> "OracleGaussianAdapter":
        self.sigma0_diag = self.sigma0_diag.to(device=device, dtype=dtype)
        return self

    def _sigma_post_diag(self, gamma: float) -> torch.Tensor:
        # (Sigma0^{-1} + gamma I)^{-1} for diagonal Sigma0
        return 1.0 / (1.0 / self.sigma0_diag + gamma)

    def denoiser(self, batch: AdapterBatch) -> torch.Tensor:
        sigma_post = self._sigma_post_diag(batch.gamma)  # (d,)
        return math.sqrt(batch.gamma) * batch.y * sigma_post[None, :]

    def jvp_denoiser(self, batch: AdapterBatch, v: torch.Tensor) -> torch.Tensor:
        # Jacobian is diagonal: J = sqrt(gamma) * diag(sigma_post)
        sigma_post = self._sigma_post_diag(batch.gamma)
        return math.sqrt(batch.gamma) * v * sigma_post[None, :]

