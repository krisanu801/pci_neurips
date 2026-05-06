from __future__ import annotations

from pathlib import Path

from pci.plots.mpl_setup import setup_matplotlib_headless

setup_matplotlib_headless()

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from pci.plots.style import set_style  # noqa: E402


def _save(fig, out_dir: Path, name: str, formats: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir / f"{name}.{fmt}", bbox_inches="tight")


def make_fig_07(spectral: pd.DataFrame, out_dir: Path, formats: list[str], top_ranks: int = 5) -> None:
    set_style()
    df = spectral.copy()
    if df.empty:
        return
    df = df[df["eig_rank"] <= top_ranks]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    sns.lineplot(data=df, x="gamma", y="eig_value", hue="model_family", style="eig_rank", ax=ax, errorbar="ci")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Fig.7: cross-family spectral flow (mean over samples)")
    ax.set_xlabel("gamma (SNR)")
    ax.set_ylabel("eigenvalue")
    _save(fig, out_dir, "fig_07", formats)
    plt.close(fig)


def make_fig_08(pairwise: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    set_style()
    df = pairwise[pairwise["stat_name"] == "mean_eigs_l2"].copy()
    if df.empty:
        return
    df["pair"] = df["family_a"].astype(str) + " vs " + df["family_b"].astype(str)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    sns.lineplot(data=df, x="gamma", y="stat_value", hue="pair", ax=ax, errorbar=None)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Fig.8: pairwise discrepancy vs gamma (L2 of mean top-k eigs)")
    ax.set_xlabel("gamma (SNR)")
    ax.set_ylabel("discrepancy")
    _save(fig, out_dir, "fig_08", formats)
    plt.close(fig)


def make_fig_10(metrics: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    set_style()
    df = metrics[metrics["metric_name"] == "trace"].copy()
    if df.empty:
        df = metrics[metrics["metric_name"] == "trace_mean"].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    sns.lineplot(data=df, x="gamma", y="metric_value", hue="model_family", ax=ax, errorbar="ci")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Fig.10: trace(Σ) vs gamma across families")
    ax.set_xlabel("gamma (SNR)")
    ax.set_ylabel("trace")
    _save(fig, out_dir, "fig_10", formats)
    plt.close(fig)


def make_all_07_to_10(run_dir: Path, formats: list[str]) -> None:
    tables = run_dir / "tables"
    figs = run_dir / "figures"
    spectral_path = tables / "spectral.csv"
    pairwise_path = tables / "pairwise.csv"
    metrics_path = tables / "metrics.csv"
    if spectral_path.exists():
        spectral = pd.read_csv(spectral_path)
        if (spectral["experiment"] == "exp2").any():
            make_fig_07(spectral[spectral["experiment"] == "exp2"], figs, formats)
    if pairwise_path.exists():
        pairwise = pd.read_csv(pairwise_path)
        make_fig_08(pairwise, figs, formats)
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        if (metrics["experiment"] == "exp2").any():
            make_fig_10(metrics[metrics["experiment"] == "exp2"], figs, formats)

