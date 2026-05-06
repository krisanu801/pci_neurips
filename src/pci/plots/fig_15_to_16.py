from __future__ import annotations

from pathlib import Path

from pci.plots.mpl_setup import setup_matplotlib_headless

setup_matplotlib_headless()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from pci.plots.style import set_style  # noqa: E402


def _save(fig, out_dir: Path, name: str, formats: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir / f"{name}.{fmt}", bbox_inches="tight")


def make_fig_16(metrics: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    set_style()
    df = metrics[metrics["metric_name"] == "mean_eigs_l2_error"].copy()
    if df.empty:
        return
    # order checkpoints by id string (step_000, step_001, ...)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    sns.lineplot(data=df, x="checkpoint_id", y="metric_value", hue="gamma", ax=ax, errorbar=None)
    ax.set_yscale("log")
    ax.set_title("Fig.16: discrepancy vs training (proxy) (mean eig L2 error)")
    ax.set_xlabel("checkpoint")
    ax.set_ylabel("L2 error")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, out_dir, "fig_16", formats)
    plt.close(fig)


def make_fig_15(spectral: pd.DataFrame, out_dir: Path, formats: list[str], ranks: list[int] | None = None) -> None:
    set_style()
    if ranks is None:
        ranks = [1, 2, 5, 10, 20]
    df = spectral.copy()
    if df.empty:
        return
    df = df[df["eig_rank"].isin(ranks)]
    # Aggregate across samples for each checkpoint/gamma/rank
    agg = (
        df.groupby(["checkpoint_id", "gamma", "eig_rank"], as_index=False)["eig_value"]
        .mean()
        .rename(columns={"eig_value": "eig_value_mean"})
    )
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    sns.lineplot(data=agg, x="checkpoint_id", y="eig_value_mean", hue="eig_rank", style="gamma", ax=ax)
    ax.set_yscale("log")
    ax.set_title("Fig.15: eigenmode convergence across checkpoints (mean eigenvalues)")
    ax.set_xlabel("checkpoint")
    ax.set_ylabel("mean eigenvalue")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, out_dir, "fig_15", formats)
    plt.close(fig)


def make_all_15_to_16(run_dir: Path, formats: list[str]) -> None:
    tables = run_dir / "tables"
    figs = run_dir / "figures"
    spectral_path = tables / "spectral.csv"
    metrics_path = tables / "metrics.csv"
    if spectral_path.exists():
        spectral = pd.read_csv(spectral_path)
        if (spectral["experiment"] == "exp4").any():
            make_fig_15(spectral[spectral["experiment"] == "exp4"], figs, formats)
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        if (metrics["experiment"] == "exp4").any():
            make_fig_16(metrics[metrics["experiment"] == "exp4"], figs, formats)

