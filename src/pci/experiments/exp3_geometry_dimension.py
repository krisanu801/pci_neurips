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
from pci.math.channel import GammaGrid
from pci.math.eigs import effective_dimension_participation_ratio, topk_eigs_exact
from pci.math.posterior import DiscreteMixture, posterior_moments_discrete
from pci.plots.make_all import maybe_make_all_figures


def _make_swiss_roll(n: int, *, device: torch.device, dtype: torch.dtype, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (x0, tangent_basis) for a 2D swiss roll embedded in R^3.
    tangent_basis: (n, 3, 2) orthonormal columns.
    """
    t = (3.0 * math.pi / 2.0) * (1.0 + 2.0 * torch.rand((n,), device=device, dtype=dtype, generator=generator))
    h = 2.0 * (torch.rand((n,), device=device, dtype=dtype, generator=generator) - 0.5)
    x = torch.stack([t * torch.cos(t), h, t * torch.sin(t)], dim=1)

    # Tangents: d/dt and d/dh
    dt = torch.stack([torch.cos(t) - t * torch.sin(t), torch.zeros_like(t), torch.sin(t) + t * torch.cos(t)], dim=1)
    dh = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).view(1, 3).repeat(n, 1)
    # Orthonormalize
    e1 = dt / (torch.norm(dt, dim=1, keepdim=True) + 1e-12)
    dh_proj = dh - torch.sum(dh * e1, dim=1, keepdim=True) * e1
    e2 = dh_proj / (torch.norm(dh_proj, dim=1, keepdim=True) + 1e-12)
    basis = torch.stack([e1, e2], dim=2)  # (n,3,2)
    return x, basis


def run(config: dict[str, Any]) -> Path:
    run_id, run_dir, events, logger = init_run(config, experiment="exp3")

    dev, dt = resolve_device(DeviceSpec(**config["device"]))
    seed = int(config["run"]["seed"])
    gen = torch.Generator(device=dev).manual_seed(seed)

    gamma_cfg = GammaGrid(**config["gamma_grid"])
    gammas_np = gamma_cfg.values()

    geom_cfg = config["geometry"]
    dataset = str(geom_cfg.get("dataset", "swiss_roll"))
    n = int(geom_cfg.get("num_samples", 512))
    ambient_dim = int(geom_cfg.get("ambient_dim", 3))

    if dataset != "swiss_roll" or ambient_dim != 3:
        raise NotImplementedError("Exp3 currently implements dataset=swiss_roll in ambient_dim=3.")

    # Discrete empirical prior on manifold support points (uniform weights).
    x_support, _ = _make_swiss_roll(max(2048, n), device=dev, dtype=dt, generator=gen)
    mix = DiscreteMixture(
        points=x_support,
        log_weights=torch.full((x_support.shape[0],), -math.log(x_support.shape[0]), device=dev, dtype=dt),
    )

    x0, tangent = _make_swiss_roll(n, device=dev, dtype=dt, generator=gen)

    spectral_path = run_dir / "tables" / "spectral.csv"
    metrics_path = run_dir / "tables" / "metrics.csv"
    geometry_path = run_dir / "tables" / "geometry.csv"

    top_k = int(config["eigs"]["top_k"])
    intrinsic_dim = 2  # swiss roll

    for g in tqdm(gammas_np, desc="exp3:gamma"):
        y = torch.sqrt(torch.tensor(float(g), device=dev, dtype=dt)) * x0 + torch.randn(x0.shape, device=dev, dtype=dt, generator=gen)
        _, cov = posterior_moments_discrete(y, float(g), mix)  # (N,3,3)

        # eigs/eigvecs per sample
        evals, evecs = torch.linalg.eigh(cov)  # evals asc, evecs columns
        evals_desc = torch.flip(evals, dims=[1])
        evecs_desc = torch.flip(evecs, dims=[2])

        topk = evals_desc[:, :top_k].detach().cpu().numpy()
        # d_eff from top-k (ambient=3 so top_k>=3 is enough)
        d_eff = effective_dimension_participation_ratio(evals_desc[:, :max(top_k, ambient_dim)]).detach().cpu().numpy()

        # Tangent overlap: projection of true tangent basis onto span of top intrinsic_dim eigenvectors.
        U = evecs_desc[:, :, :intrinsic_dim]  # (N,3,2)
        # overlap = || T^T U ||_F^2 / intrinsic_dim  (T: 3x2 orthonormal)
        TTU = torch.einsum("nab,nbc->nac", tangent.transpose(1, 2), U)  # (N,2,2)
        overlap = (TTU * TTU).sum(dim=(1, 2)) / intrinsic_dim
        overlap_np = overlap.detach().cpu().numpy()

        # Off-tangent energy proxy: fraction of trace outside top intrinsic_dim
        tr = torch.sum(evals_desc, dim=1)
        off = torch.sum(evals_desc[:, intrinsic_dim:], dim=1)
        off_frac = (off / (tr + 1e-12)).detach().cpu().numpy()

        for i in range(n):
            df_s = pd.DataFrame(
                [{
                    "run_id": run_id,
                    "experiment": "exp3",
                    "dataset": dataset,
                    "model_family": "exact_discrete",
                    "checkpoint_id": "n/a",
                    "sample_id": i,
                    "gamma": float(g),
                    "eig_rank": int(r + 1),
                    "eig_value": float(val),
                } for r, val in enumerate(topk[i])]
            )
            append_csv(df_s, spectral_path)

            df_g = pd.DataFrame(
                [{
                    "run_id": run_id,
                    "experiment": "exp3",
                    "dataset": dataset,
                    "sample_id": i,
                    "gamma": float(g),
                    "d_eff": float(d_eff[i]),
                    "tangent_overlap": float(overlap_np[i]),
                    "off_tangent_energy": float(off_frac[i]),
                }]
            )
            append_csv(df_g, geometry_path)

            df_m = pd.DataFrame(
                [{
                    "run_id": run_id,
                    "experiment": "exp3",
                    "dataset": dataset,
                    "model_family": "exact_discrete",
                    "checkpoint_id": "n/a",
                    "sample_id": i,
                    "gamma": float(g),
                    "metric_name": "trace",
                    "metric_value": float(tr[i].item()),
                }]
            )
            append_csv(df_m, metrics_path)

    if config.get("plots", {}).get("make_figures", False):
        maybe_make_all_figures(run_dir, formats=config["plots"].get("formats", ["pdf", "png"]))
    events.log("run_end", run_id=run_id, experiment="exp3")
    return run_dir


def main() -> None:
    cfg = parse_config_arg()
    run_dir = run(cfg)
    print(str(run_dir.resolve()))


if __name__ == "__main__":
    main()
