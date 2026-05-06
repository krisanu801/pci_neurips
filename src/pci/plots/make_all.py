from __future__ import annotations

import argparse
from pathlib import Path

from pci.logging_utils import setup_console_logger
from pci.plots.fig_02_to_06 import make_all_02_to_06
from pci.plots.fig_07_to_10 import make_all_07_to_10
from pci.plots.fig_11_to_14 import make_all_11_to_14
from pci.plots.fig_15_to_16 import make_all_15_to_16


def maybe_make_all_figures(run_dir: Path, formats: list[str]) -> None:
    logger = setup_console_logger()
    logger.info(f"[plots] generating figures from CSV in: {run_dir}")
    make_all_02_to_06(run_dir, formats=formats)
    make_all_07_to_10(run_dir, formats=formats)
    make_all_11_to_14(run_dir, formats=formats)
    make_all_15_to_16(run_dir, formats=formats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=str)
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    args = ap.parse_args()
    maybe_make_all_figures(Path(args.run_dir), formats=list(args.formats))


if __name__ == "__main__":
    main()
