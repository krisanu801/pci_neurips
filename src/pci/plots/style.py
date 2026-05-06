from __future__ import annotations

from pci.plots.mpl_setup import setup_matplotlib_headless

setup_matplotlib_headless()

import matplotlib as mpl  # noqa: E402
import seaborn as sns  # noqa: E402


def set_style() -> None:
    sns.set_theme(style="whitegrid")
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
