from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping, got: {type(data)}")
    return data


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


@dataclass(frozen=True)
class ResolvedRunPaths:
    run_dir: Path
    tables_dir: Path
    figures_dir: Path


def ensure_run_dirs(out_dir: str | Path, run_id: str) -> ResolvedRunPaths:
    base = Path(out_dir) / run_id
    tables = base / "tables"
    figures = base / "figures"
    base.mkdir(parents=True, exist_ok=False)
    tables.mkdir(parents=True, exist_ok=False)
    figures.mkdir(parents=True, exist_ok=False)
    return ResolvedRunPaths(run_dir=base, tables_dir=tables, figures_dir=figures)

