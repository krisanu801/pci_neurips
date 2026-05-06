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


def make_fig_11(geo: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    sns.lineplot(data=geo, x="gamma", y="d_eff", hue="dataset", ax=ax, errorbar="ci")
    ax.set_xscale("log")
    ax.set_title("Fig.11: effective dimension d_eff(gamma)")
    ax.set_xlabel("gamma (SNR)")
    ax.set_ylabel("d_eff")
    _save(fig, out_dir, "fig_11", formats)
    plt.close(fig)


def make_fig_13(geo: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    if "tangent_overlap" not in geo.columns:
        return
    set_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    sns.lineplot(data=geo, x="gamma", y="tangent_overlap", hue="dataset", ax=ax, errorbar="ci")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fig.13: tangent-space overlap vs gamma")
    ax.set_xlabel("gamma (SNR)")
    ax.set_ylabel("overlap")
    _save(fig, out_dir, "fig_13", formats)
    plt.close(fig)


def make_fig_14(geo: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    if "off_tangent_energy" not in geo.columns:
        return
    set_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    sns.lineplot(data=geo, x="gamma", y="off_tangent_energy", hue="dataset", ax=ax, errorbar="ci")
    ax.set_xscale("log")
    ax.set_title("Fig.14: off-tangent energy fraction vs gamma")
    ax.set_xlabel("gamma (SNR)")
    ax.set_ylabel("fraction")
    _save(fig, out_dir, "fig_14", formats)
    plt.close(fig)


def make_all_11_to_14(run_dir: Path, formats: list[str]) -> None:
    tables = run_dir / "tables"
    figs = run_dir / "figures"
    geo_path = tables / "geometry.csv"
    if not geo_path.exists():
        return
    geo = pd.read_csv(geo_path)
    if not (geo["experiment"] == "exp3").any():
        return
    geo = geo[geo["experiment"] == "exp3"]
    make_fig_11(geo, figs, formats)
    make_fig_13(geo, figs, formats)
    make_fig_14(geo, figs, formats)

