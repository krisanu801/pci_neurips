from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pci.device import DeviceSpec, resolve_device
from pci.experiments.common import init_run, parse_config_arg
from pci.io.write import append_csv
from pci.math.channel import GammaGrid, corrupt_standardized
from pci.math.power_eigs import PowerSpec, topk_eigs_power
from pci.models.adapters import OracleGaussianAdapter, StubAdapter
from pci.models.adapters.base import AdapterBatch, ModelAdapter
from pci.plots.make_all import maybe_make_all_figures


def _build_adapter(model_cfg: dict[str, Any], *, dim: int, device: torch.device, dtype: torch.dtype) -> ModelAdapter:
    adapter_type = model_cfg.get("adapter_type", "stub")
    family = str(model_cfg.get("model_family", adapter_type))
    if adapter_type == "oracle_gaussian":
        sigma0_diag = torch.logspace(math.log10(0.2), math.log10(5.0), dim)
        return OracleGaussianAdapter(model_family=family, sigma0_diag=sigma0_diag).to(device, dtype)
    if adapter_type == "stub":
        return StubAdapter(model_family=family).to(device, dtype)
    raise ValueError(f"Unknown adapter_type={adapter_type}")


def run(config: dict[str, Any]) -> Path:
    run_id, run_dir, events, logger = init_run(config, experiment="exp2")

    dev, dt = resolve_device(DeviceSpec(**config["device"]))
    seed = int(config["run"]["seed"])
    gen = torch.Generator(device=dev).manual_seed(seed)

    gamma_cfg = GammaGrid(**config["gamma_grid"])
    gammas_np = gamma_cfg.values()
    gammas = torch.tensor(gammas_np, device=dev, dtype=dt)

    data_cfg = config["data"]
    dataset = str(data_cfg.get("dataset", "random_gaussian"))
    n = int(data_cfg.get("num_samples", 128))
    dim = int(data_cfg.get("dim", 256))
    if dataset != "random_gaussian":
        raise NotImplementedError("Exp2 currently ships with dataset=random_gaussian for pipeline validation.")

    # For pipeline validation, use a Gaussian synthetic x0 (this exercises all CSV/plot code).
    x0 = torch.randn((n, dim), device=dev, dtype=dt, generator=gen)

    eig_cfg = config["eigs"]
    top_k = int(eig_cfg["top_k"])
    power_spec = PowerSpec(iters=int(eig_cfg.get("power_iters", 120)), tol=float(eig_cfg.get("power_tol", 1e-6)))

    spectral_path = run_dir / "tables" / "spectral.csv"
    metrics_path = run_dir / "tables" / "metrics.csv"
    pairwise_path = run_dir / "tables" / "pairwise.csv"

    adapters: list[ModelAdapter] = []
    for mcfg in config.get("models", []):
        adapters.append(_build_adapter(mcfg, dim=dim, device=dev, dtype=dt))

    families = [a.model_family for a in adapters]
    logger.info(f"[exp2] dataset={dataset} n={n} dim={dim} families={families}")

    # Main loop: for each gamma and sample, extract top-k eigenvalues of Sigma_hat = (1/sqrt(gamma)) * J_D
    for g in tqdm(gammas_np, desc="exp2:gamma"):
        y = corrupt_standardized(x0, torch.tensor(float(g), device=dev, dtype=dt), generator=gen)  # (N,d)

        mean_eigs_by_family: dict[str, np.ndarray] = {}
        for adapter in adapters:
            batch = AdapterBatch(y=y, gamma=float(g))

            # Per-sample matvec uses the adapter's JVP.
            eigs_all = []
            for i in range(n):
                yi = batch.y[i : i + 1, :]
                bi = AdapterBatch(y=yi, gamma=float(g))

                def matvec(v: torch.Tensor) -> torch.Tensor:
                    v2 = v.view(1, -1)
                    jv = adapter.jvp_denoiser(bi, v2)  # (1,d)
                    return (jv / math.sqrt(float(g))).view(-1)

                evals = topk_eigs_power(
                    matvec,
                    d=dim,
                    k=top_k,
                    device=dev,
                    dtype=dt,
                    spec=power_spec,
                    generator=gen,
                ).detach().cpu().numpy()
                eigs_all.append(evals)

                df = pd.DataFrame(
                    [{
                        "run_id": run_id,
                        "experiment": "exp2",
                        "dataset": dataset,
                        "model_family": adapter.model_family,
                        "checkpoint_id": "pretrained",
                        "sample_id": i,
                        "gamma": float(g),
                        "eig_rank": int(r + 1),
                        "eig_value": float(val),
                    } for r, val in enumerate(evals)]
                )
                append_csv(df, spectral_path)

                dfm = pd.DataFrame(
                    [{
                        "run_id": run_id,
                        "experiment": "exp2",
                        "dataset": dataset,
                        "model_family": adapter.model_family,
                        "checkpoint_id": "pretrained",
                        "sample_id": i,
                        "gamma": float(g),
                        "metric_name": "trace",
                        "metric_value": float(np.sum(evals)),
                    }]
                )
                append_csv(dfm, metrics_path)

            eigs_all = np.stack(eigs_all, axis=0)  # (N,k)
            mean_eigs_by_family[adapter.model_family] = eigs_all.mean(axis=0)

        # Pairwise discrepancies between mean spectral curves at this gamma.
        fams = list(mean_eigs_by_family.keys())
        for i in range(len(fams)):
            for j in range(i + 1, len(fams)):
                a, b = fams[i], fams[j]
                da = mean_eigs_by_family[a]
                db = mean_eigs_by_family[b]
                l2 = float(np.linalg.norm(da - db))
                dfp = pd.DataFrame(
                    [{
                        "run_id": run_id,
                        "experiment": "exp2",
                        "dataset": dataset,
                        "gamma": float(g),
                        "family_a": a,
                        "family_b": b,
                        "stat_name": "mean_eigs_l2",
                        "stat_value": l2,
                    }]
                )
                append_csv(dfp, pairwise_path)

    if config.get("plots", {}).get("make_figures", False):
        maybe_make_all_figures(run_dir, formats=config["plots"].get("formats", ["pdf", "png"]))
    events.log("run_end", run_id=run_id, experiment="exp2")
    return run_dir


def main() -> None:
    cfg = parse_config_arg()
    run_dir = run(cfg)
    print(str(run_dir.resolve()))


if __name__ == "__main__":
    main()
