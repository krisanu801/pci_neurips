from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def append_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        df.to_csv(path, index=False)
        return
    df.to_csv(path, mode="a", header=False, index=False)

