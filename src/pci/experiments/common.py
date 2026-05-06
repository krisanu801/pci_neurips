from __future__ import annotations

import argparse
import datetime as _dt
import json
import secrets
from pathlib import Path
from typing import Any

from pci.config import load_yaml, ensure_run_dirs
from pci.logging_utils import JsonlLogger, setup_console_logger
from pci.manifest import write_manifest


def make_run_id(prefix: str) -> str:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    salt = secrets.token_hex(3)
    return f"{prefix}_{ts}_{salt}"


def parse_config_arg(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    args = ap.parse_args(argv)
    return load_yaml(args.config)


def init_run(config: dict[str, Any], *, experiment: str) -> tuple[str, Path, JsonlLogger, Any]:
    logger = setup_console_logger()
    run_id = make_run_id(config.get("run", {}).get("name", experiment))
    out_root = Path(config.get("run", {}).get("out_dir", "runs"))
    paths = ensure_run_dirs(out_root, run_id)
    events = JsonlLogger(paths.run_dir / "events.jsonl")
    write_manifest(paths.run_dir / "manifest.json", config)
    logger.info(f"run_id={run_id}")
    logger.info(f"run_dir={paths.run_dir.resolve()}")
    events.log("run_start", run_id=run_id, experiment=experiment)
    return run_id, paths.run_dir, events, logger

