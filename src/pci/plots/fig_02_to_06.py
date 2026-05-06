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


def make_fig_02(metrics: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    set_style()
    df = metrics[metrics["metric_name"] == "jac_residual_fro"].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    sns.lineplot(data=df, x="gamma", y="metric_value", hue="dataset", style="model_family", ax=ax, errorbar="ci")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Fig.2: ||grad_y D - sqrt(gamma) * Sigma||_F vs gamma (oracle)")
    ax.set_xlabel("gamma (SNR)")
    ax.set_ylabel("Frobenius residual")
    _save(fig, out_dir, "fig_02", formats)
    plt.close(fig)


def make_fig_03(metrics: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    set_style()
    df = metrics[metrics["metric_name"].isin(["trace", "sigma_from_jac_fro_error"])].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    sns.lineplot(data=df, x="gamma", y="metric_value", hue="metric_name", style="dataset", ax=ax, errorbar="ci")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Fig.3: trace(Σ) and ||Σ̂(J)−Σ||_F vs γ (synthetic)")
    ax.set_xlabel("gamma (SNR)")
    ax.set_ylabel("value")
    _save(fig, out_dir, "fig_03", formats)
    plt.close(fig)


def make_fig_05(spectral: pd.DataFrame, out_dir: Path, formats: list[str], top_ranks: int = 8) -> None:
    set_style()
    df = spectral.copy()
    if df.empty:
        return
    df = df[df["eig_rank"] <= top_ranks]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    sns.lineplot(data=df, x="gamma", y="eig_value", hue="eig_rank", style="dataset", ax=ax, errorbar=None)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Fig.5: top-{top_ranks} eigenvalues vs gamma")
    ax.set_xlabel("gamma (SNR)")
    ax.set_ylabel("eigenvalue")
    _save(fig, out_dir, "fig_05", formats)
    plt.close(fig)


def make_all_02_to_06(run_dir: Path, formats: list[str]) -> None:
    tables = run_dir / "tables"
    figs = run_dir / "figures"
    metrics_path = tables / "metrics.csv"
    spectral_path = tables / "spectral.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        if (metrics.get("experiment") == "exp1").any():
            m1 = metrics[metrics["experiment"] == "exp1"]
            make_fig_02(m1, figs, formats)
            make_fig_03(m1, figs, formats)
    if spectral_path.exists():
        spectral = pd.read_csv(spectral_path)
        if (spectral.get("experiment") == "exp1").any():
            s1 = spectral[spectral["experiment"] == "exp1"]
            make_fig_05(s1, figs, formats)
