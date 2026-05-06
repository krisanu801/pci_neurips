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
from pci.models.adapters.base import AdapterBatch, ModelAdapter
from pci.models.adapters.oracle_gaussian import OracleGaussianAdapter
from pci.plots.make_all import maybe_make_all_figures


class NoisyJvpAdapter(ModelAdapter):
    """
    Simple dynamics simulator: wraps a base adapter, but adds Gaussian noise to JVPs.
    Noise decays across checkpoints -> eigenmode errors decay across training.
    """

    def __init__(self, base: ModelAdapter, *, noise_std: float, generator: torch.Generator):
        self.base = base
        self.model_family = getattr(base, "model_family", "base")
        self.noise_std = float(noise_std)
        self._gen = generator

    def to(self, device: torch.device, dtype: torch.dtype) -> "NoisyJvpAdapter":
        self.base.to(device, dtype)
        return self

    def denoiser(self, batch: AdapterBatch) -> torch.Tensor:
        return self.base.denoiser(batch)

    def jvp_denoiser(self, batch: AdapterBatch, v: torch.Tensor) -> torch.Tensor:
        jv = self.base.jvp_denoiser(batch, v)
        if self.noise_std <= 0:
            return jv
        noise = torch.randn(jv.shape, device=jv.device, dtype=jv.dtype, generator=self._gen) * self.noise_std
        return jv + noise


def run(config: dict[str, Any]) -> Path:
    run_id, run_dir, events, logger = init_run(config, experiment="exp4")

    dev, dt = resolve_device(DeviceSpec(**config["device"]))
    seed = int(config["run"]["seed"])
    gen = torch.Generator(device=dev).manual_seed(seed)

    gamma_cfg = GammaGrid(**config["gamma_grid"])
    gammas_np = gamma_cfg.values()

    eig_cfg = config["eigs"]
    top_k = int(eig_cfg["top_k"])
    power_spec = PowerSpec(iters=int(eig_cfg.get("power_iters", 120)), tol=float(eig_cfg.get("power_tol", 1e-6)))

    mode = str(config.get("mode", "evaluate_checkpoints"))
    ckpt_cfg = config.get("checkpoints", {})
    checkpoint_ids = list(ckpt_cfg.get("checkpoint_ids", []))

    simulate = (mode != "evaluate_checkpoints") or (len(checkpoint_ids) == 0)
    if not simulate:
        raise NotImplementedError("Exp4 evaluate_checkpoints is not implemented yet. Use mode=train_toy (simulation).")

    # Synthetic dataset for dynamics: random Gaussian x0 in R^d.
    dim = int(config.get("data", {}).get("dim", 256))
    n = int(config.get("data", {}).get("num_samples", 64))
    x0 = torch.randn((n, dim), device=dev, dtype=dt, generator=gen)

    sigma0_diag = torch.logspace(math.log10(0.2), math.log10(5.0), dim, device=dev, dtype=dt)
    base = OracleGaussianAdapter(model_family="oracle", sigma0_diag=sigma0_diag).to(dev, dt)

    probe_cfg = config.get("probe", {})
    ckpt_count = int(probe_cfg.get("checkpoints_count", 11))
    ckpt_count = max(2, ckpt_count)
    checkpoint_ids = [f"step_{i:03d}" for i in range(0, ckpt_count)]
    noise_stds = np.geomspace(1.0, 0.01, num=len(checkpoint_ids))
    adapters = [NoisyJvpAdapter(base, noise_std=float(s), generator=gen).to(dev, dt) for s in noise_stds]
    val_losses = (noise_stds**2).tolist()
    logger.info(f"[exp4] simulate checkpoints={len(checkpoint_ids)} dim={dim} n={n}")

    spectral_path = run_dir / "tables" / "spectral.csv"
    metrics_path = run_dir / "tables" / "metrics.csv"

    probe_count = int(probe_cfg.get("gamma_probe_count", 3))
    probe_count = max(1, min(probe_count, len(gammas_np)))
    idxs = np.linspace(0, len(gammas_np) - 1, num=probe_count, dtype=int).tolist()
    gamma_probe = [float(gammas_np[i]) for i in idxs]

    for ckpt_i, ckpt_id in enumerate(checkpoint_ids):
        adapter = adapters[ckpt_i]

        # checkpoint-level scalar (synthetic proxy)
        dfv = pd.DataFrame(
            [{
                "run_id": run_id,
                "experiment": "exp4",
                "dataset": "random_gaussian",
                "model_family": "oracle_noisy",
                "checkpoint_id": ckpt_id,
                "sample_id": -1,
                "gamma": float("nan"),
                "metric_name": "val_loss_proxy",
                "metric_value": float(val_losses[ckpt_i]),
            }]
        )
        append_csv(dfv, metrics_path)

        for g in tqdm(gamma_probe, desc=f"exp4:{ckpt_id}", leave=False):
            y = corrupt_standardized(x0, torch.tensor(g, device=dev, dtype=dt), generator=gen)
            batch = AdapterBatch(y=y, gamma=float(g))

            # Analytic ground-truth spectrum for the Gaussian prior (for error plots).
            sigma_post = (1.0 / (1.0 / sigma0_diag + float(g))).detach().cpu().numpy()
            true_eigs = np.sort(sigma_post)[::-1][:top_k]

            eigs_all = []
            for i in range(n):
                yi = batch.y[i : i + 1, :]
                bi = AdapterBatch(y=yi, gamma=float(g))

                def matvec(v: torch.Tensor) -> torch.Tensor:
                    v2 = v.view(1, -1)
                    jv = adapter.jvp_denoiser(bi, v2)
                    return (jv / math.sqrt(float(g))).view(-1)

                evals = topk_eigs_power(matvec, d=dim, k=top_k, device=dev, dtype=dt, spec=power_spec, generator=gen)
                eigs = evals.detach().cpu().numpy()
                eigs_all.append(eigs)

                df = pd.DataFrame(
                    [{
                        "run_id": run_id,
                        "experiment": "exp4",
                        "dataset": "random_gaussian",
                        "model_family": "oracle_noisy",
                        "checkpoint_id": ckpt_id,
                        "sample_id": i,
                        "gamma": float(g),
                        "eig_rank": int(r + 1),
                        "eig_value": float(val),
                    } for r, val in enumerate(eigs)]
                )
                append_csv(df, spectral_path)

            eigs_all = np.stack(eigs_all, axis=0)
            mean_eigs = eigs_all.mean(axis=0)
            mean_trace = float(eigs_all.mean(axis=0).sum())
            dfm = pd.DataFrame(
                [{
                    "run_id": run_id,
                    "experiment": "exp4",
                    "dataset": "random_gaussian",
                    "model_family": "oracle_noisy",
                    "checkpoint_id": ckpt_id,
                    "sample_id": -1,
                    "gamma": float(g),
                    "metric_name": "trace_mean",
                    "metric_value": mean_trace,
                }]
            )
            append_csv(dfm, metrics_path)

            # Spectral error vs truth (mean eigenvalues).
            err_l2 = float(np.linalg.norm(mean_eigs - true_eigs))
            dfe = pd.DataFrame(
                [{
                    "run_id": run_id,
                    "experiment": "exp4",
                    "dataset": "random_gaussian",
                    "model_family": "oracle_noisy",
                    "checkpoint_id": ckpt_id,
                    "sample_id": -1,
                    "gamma": float(g),
                    "metric_name": "mean_eigs_l2_error",
                    "metric_value": err_l2,
                }]
            )
            append_csv(dfe, metrics_path)

    if config.get("plots", {}).get("make_figures", False):
        maybe_make_all_figures(run_dir, formats=config["plots"].get("formats", ["pdf", "png"]))

    events.log("run_end", run_id=run_id, experiment="exp4")
    return run_dir


def main() -> None:
    cfg = parse_config_arg()
    run_dir = run(cfg)
    print(str(run_dir.resolve()))


if __name__ == "__main__":
    main()
