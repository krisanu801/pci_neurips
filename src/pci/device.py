from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceSpec:
    device: str  # "auto"|"cpu"|"cuda"
    dtype: str   # "float32"|"float64"


def resolve_device(spec: DeviceSpec) -> tuple[torch.device, torch.dtype]:
    if spec.device == "auto":
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        dev = torch.device(spec.device)

    if spec.dtype == "float32":
        dt = torch.float32
    elif spec.dtype == "float64":
        dt = torch.float64
    else:
        raise ValueError(f"Unsupported dtype: {spec.dtype}")

    return dev, dt

