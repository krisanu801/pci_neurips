from __future__ import annotations

from dataclasses import dataclass

import torch

from pci.models.adapters.base import AdapterBatch, ModelAdapter


@dataclass
class StubAdapter(ModelAdapter):
    model_family: str = "stub"

    def to(self, device: torch.device, dtype: torch.dtype) -> "StubAdapter":
        return self

    def denoiser(self, batch: AdapterBatch) -> torch.Tensor:
        raise NotImplementedError(
            "StubAdapter cannot run. Provide a real adapter under src/pci/models/adapters/ "
            "that implements denoiser() and jvp_denoiser()."
        )

    def jvp_denoiser(self, batch: AdapterBatch, v: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "StubAdapter cannot run. Provide a real adapter under src/pci/models/adapters/."
        )

