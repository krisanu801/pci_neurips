from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def _maybe_git_hash() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def write_manifest(path: Path, config: dict[str, Any]) -> None:
    data = {
        "config": config,
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "git_hash": _maybe_git_hash(),
            "torch": getattr(torch, "__version__", None),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": getattr(torch.version, "cuda", None),
        },
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

