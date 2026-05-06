from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class CsvSchema:
    name: str
    required_columns: tuple[str, ...]

    def validate_df(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"{self.name}: missing columns: {missing}")


SPECTRAL_SCHEMA = CsvSchema(
    name="spectral.csv",
    required_columns=(
        "run_id",
        "experiment",
        "dataset",
        "model_family",
        "checkpoint_id",
        "sample_id",
        "gamma",
        "eig_rank",
        "eig_value",
    ),
)

METRICS_SCHEMA = CsvSchema(
    name="metrics.csv",
    required_columns=(
        "run_id",
        "experiment",
        "dataset",
        "model_family",
        "checkpoint_id",
        "sample_id",
        "gamma",
        "metric_name",
        "metric_value",
    ),
)

PAIRWISE_SCHEMA = CsvSchema(
    name="pairwise.csv",
    required_columns=(
        "run_id",
        "experiment",
        "dataset",
        "gamma",
        "family_a",
        "family_b",
        "stat_name",
        "stat_value",
    ),
)

GEOMETRY_SCHEMA = CsvSchema(
    name="geometry.csv",
    required_columns=(
        "run_id",
        "experiment",
        "dataset",
        "sample_id",
        "gamma",
        "d_eff",
    ),
)


def validate_csv(path: str) -> None:
    df = pd.read_csv(path)
    name = path.split("/")[-1]
    schema = {
        "spectral.csv": SPECTRAL_SCHEMA,
        "metrics.csv": METRICS_SCHEMA,
        "pairwise.csv": PAIRWISE_SCHEMA,
        "geometry.csv": GEOMETRY_SCHEMA,
    }.get(name)
    if schema is None:
        raise ValueError(f"Unknown CSV schema for: {name}")
    schema.validate_df(df)

