from __future__ import annotations

import math
from dataclasses import dataclass
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
from pci.math.eigs import EigsSpec, topk_eigs_exact
from pci.math.posterior import (
    DiscreteMixture,
    GaussianMixture,
    posterior_cov_gaussian_prior,
    posterior_moments_discrete,
    posterior_moments_gmm,
)
from pci.plots.make_all import maybe_make_all_figures


def _make_gaussian_diag_cov(dim: int, eig_min: float, eig_max: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    eigs = torch.logspace(math.log10(eig_min), math.log10(eig_max), dim, device=device, dtype=dtype)
    return torch.diag(eigs)


def _sample_gaussian(n: int, cov: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
    d = cov.shape[0]
    L = torch.linalg.cholesky(cov)
    z = torch.randn((n, d), device=cov.device, dtype=cov.dtype, generator=generator)
    return z @ L.T


def _sample_two_point(n: int, dim: int, mu_norm: float, device: torch.device, dtype: torch.dtype, *, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    v = torch.randn((dim,), device=device, dtype=dtype, generator=generator)
    v = v / (torch.norm(v) + 1e-12)
    mu = mu_norm * v
    coins = torch.randint(0, 2, (n,), device=device, generator=generator)
    x = torch.where(coins[:, None] == 0, mu[None, :], -mu[None, :])
    return x, mu


def _make_gmm_iso(dim: int, k: int, separation: float, sigma: float, device: torch.device, dtype: torch.dtype, *, generator: torch.Generator) -> GaussianMixture:
    means = []
    for _ in range(k):
        v = torch.randn((dim,), device=device, dtype=dtype, generator=generator)
        v = v / (torch.norm(v) + 1e-12)
        means.append(separation * v)
    means = torch.stack(means, dim=0)
    cov = (sigma * sigma) * torch.eye(dim, device=device, dtype=dtype)
    covs = cov.unsqueeze(0).repeat(k, 1, 1)
    weights = torch.ones((k,), device=device, dtype=dtype) / k
    return GaussianMixture(weights=weights, means=means, covs=covs)


def run(config: dict[str, Any]) -> Path:
    run_id, run_dir, events, logger = init_run(config, experiment="exp1")

    dev, dt = resolve_device(DeviceSpec(**config["device"]))
    seed = int(config["run"]["seed"])
    gen = torch.Generator(device=dev).manual_seed(seed)

    gamma_cfg = GammaGrid(**config["gamma_grid"])
    gammas_np = gamma_cfg.values()
    gammas = torch.tensor(gammas_np, device=dev, dtype=dt)

    eigs_spec = EigsSpec(**config["eigs"])
    n = int(config["synthetic"]["num_samples"])
    dim = int(config["synthetic"]["dim"])
    n_jac = int(config["synthetic"].get("num_jacobian_samples", 0))

    spectral_path = run_dir / "tables" / "spectral.csv"
    metrics_path = run_dir / "tables" / "metrics.csv"

    def emit_metric(dataset: str, model_family: str, checkpoint_id: str, sample_id: int, gamma: float, name: str, value: float) -> None:
        df = pd.DataFrame(
            [{
                "run_id": run_id,
                "experiment": "exp1",
                "dataset": dataset,
                "model_family": model_family,
                "checkpoint_id": checkpoint_id,
                "sample_id": sample_id,
                "gamma": float(gamma),
                "metric_name": name,
                "metric_value": float(value),
            }]
        )
        append_csv(df, metrics_path)

    def emit_spectrum(dataset: str, model_family: str, checkpoint_id: str, sample_id: int, gamma: float, eigvals: np.ndarray) -> None:
        df = pd.DataFrame(
            [{
                "run_id": run_id,
                "experiment": "exp1",
                "dataset": dataset,
                "model_family": model_family,
                "checkpoint_id": checkpoint_id,
                "sample_id": sample_id,
                "gamma": float(gamma),
                "eig_rank": int(i + 1),
                "eig_value": float(v),
            } for i, v in enumerate(eigvals)]
        )
        append_csv(df, spectral_path)

    for ds in config["datasets"]:
        name = ds["name"]
        params = ds.get("params", {})
        events.log("dataset_start", run_id=run_id, dataset=name)
        logger.info(f"[exp1] dataset={name}")

        if name == "gaussian_diag":
            cov0 = _make_gaussian_diag_cov(dim, float(params["eig_min"]), float(params["eig_max"]), dev, dt)
            x0 = _sample_gaussian(n, cov0, generator=gen)
            # Sigma_post is independent of y; compute once per gamma.
            Sigma_post = posterior_cov_gaussian_prior(gammas, cov0)  # (G,d,d)
            topk = topk_eigs_exact(Sigma_post, eigs_spec.top_k).detach().cpu().numpy()  # (G,k)
            for gi, g in enumerate(gammas_np):
                emit_spectrum(name, "exact", "n/a", -1, g, topk[gi])
                emit_metric(name, "exact", "n/a", -1, g, "trace", float(np.sum(topk[gi])))

                # Oracle Jacobian identity residual for Gaussian: ∇_y D = sqrt(gamma) Σ_post exactly.
                emit_metric(name, "oracle", "n/a", -1, g, "jac_residual_fro", 0.0)
                emit_metric(name, "oracle", "n/a", -1, g, "sigma_from_jac_fro_error", 0.0)

        elif name == "two_point":
            x0, mu = _sample_two_point(n, dim, float(params["mu_norm"]), dev, dt, generator=gen)
            mix = DiscreteMixture(points=torch.stack([mu, -mu], dim=0), log_weights=torch.log(torch.tensor([0.5, 0.5], device=dev, dtype=dt)))
            # Sample y once per gamma so we can compute sample-dependent posterior.
            for gi, g in enumerate(tqdm(gammas_np, desc=f"{name}:gamma")):
                y = torch.sqrt(torch.tensor(g, device=dev, dtype=dt)) * x0 + torch.randn(x0.shape, device=dev, dtype=dt, generator=gen)
                _, cov = posterior_moments_discrete(y, float(g), mix)  # (N,d,d)
                evals = topk_eigs_exact(cov, eigs_spec.top_k).detach().cpu().numpy()  # (N,k)
                for i in range(n):
                    emit_spectrum(name, "exact", "n/a", i, g, evals[i])
                    emit_metric(name, "exact", "n/a", i, g, "trace", float(evals[i].sum()))

                # Jacobian identity check on a small subset (full dxd Jacobian).
                if n_jac > 0:
                    idx = torch.arange(min(n_jac, n), device=dev)
                    y_sub = y[idx].detach().clone().requires_grad_(True)

                    def mean_fn(y_in: torch.Tensor) -> torch.Tensor:
                        m, _ = posterior_moments_discrete(y_in, float(g), mix)
                        return m

                    # Compute Jacobian per sample to keep memory predictable.
                    for j in range(y_sub.shape[0]):
                        yj = y_sub[j]
                        mj = mean_fn(yj.unsqueeze(0)).squeeze(0)
                        J = torch.autograd.functional.jacobian(lambda yy: mean_fn(yy.unsqueeze(0)).squeeze(0), yj, create_graph=False)
                        # J: (d,d)
                        sigma_hat = J / math.sqrt(float(g))
                        sigma_true = cov[idx[j]].detach()
                        resid = torch.linalg.norm(J - math.sqrt(float(g)) * sigma_true).item()
                        ferr = torch.linalg.norm(sigma_hat - sigma_true).item()
                        emit_metric(name, "oracle", "n/a", int(idx[j].item()), g, "jac_residual_fro", resid)
                        emit_metric(name, "oracle", "n/a", int(idx[j].item()), g, "sigma_from_jac_fro_error", ferr)
                        # entropy of posterior weights (phase transition signal)
                        with torch.no_grad():
                            diff = y[idx[j]].unsqueeze(0)[:, None, :] - math.sqrt(float(g)) * mix.points[None, :, :]
                            log_like = -0.5 * torch.sum(diff * diff, dim=-1)
                            lp = log_like + mix.log_weights[None, :]
                            lp = lp - torch.logsumexp(lp, dim=-1, keepdim=True)
                            w = torch.exp(lp).squeeze(0)
                            ent = float(-(w * torch.log(w + 1e-30)).sum().item())
                            emit_metric(name, "oracle", "n/a", int(idx[j].item()), g, "post_entropy", ent)

        elif name == "gmm_iso":
            gmm = _make_gmm_iso(dim, int(params["k"]), float(params["separation"]), float(params["sigma"]), dev, dt, generator=gen)
            # sample x0 from prior
            comp = torch.multinomial(gmm.weights, num_samples=n, replacement=True, generator=gen)  # (N,)
            x0 = gmm.means[comp] + torch.randn((n, dim), device=dev, dtype=dt, generator=gen) * float(params["sigma"])
            for gi, g in enumerate(tqdm(gammas_np, desc=f"{name}:gamma")):
                y = torch.sqrt(torch.tensor(g, device=dev, dtype=dt)) * x0 + torch.randn(x0.shape, device=dev, dtype=dt, generator=gen)
                _, cov = posterior_moments_gmm(y, float(g), gmm)
                evals = topk_eigs_exact(cov, eigs_spec.top_k).detach().cpu().numpy()
                for i in range(n):
                    emit_spectrum(name, "exact", "n/a", i, g, evals[i])
                    emit_metric(name, "exact", "n/a", i, g, "trace", float(evals[i].sum()))

                if n_jac > 0:
                    idx = torch.arange(min(n_jac, n), device=dev)
                    y_sub = y[idx].detach().clone().requires_grad_(True)

                    def mean_fn(y_in: torch.Tensor) -> torch.Tensor:
                        m, _ = posterior_moments_gmm(y_in, float(g), gmm)
                        return m

                    for j in range(y_sub.shape[0]):
                        yj = y_sub[j]
                        J = torch.autograd.functional.jacobian(lambda yy: mean_fn(yy.unsqueeze(0)).squeeze(0), yj, create_graph=False)
                        sigma_hat = J / math.sqrt(float(g))
                        sigma_true = cov[idx[j]].detach()
                        resid = torch.linalg.norm(J - math.sqrt(float(g)) * sigma_true).item()
                        ferr = torch.linalg.norm(sigma_hat - sigma_true).item()
                        emit_metric(name, "oracle", "n/a", int(idx[j].item()), g, "jac_residual_fro", resid)
                        emit_metric(name, "oracle", "n/a", int(idx[j].item()), g, "sigma_from_jac_fro_error", ferr)
        else:
            raise ValueError(f"Unknown dataset: {name}")

        events.log("dataset_end", run_id=run_id, dataset=name)

    events.log("run_end", run_id=run_id, experiment="exp1")

    if config.get("plots", {}).get("make_figures", False):
        maybe_make_all_figures(run_dir, formats=config["plots"].get("formats", ["pdf", "png"]))

    return run_dir


def main() -> None:
    cfg = parse_config_arg()
    run_dir = run(cfg)
    print(str(run_dir.resolve()))


if __name__ == "__main__":
    main()
