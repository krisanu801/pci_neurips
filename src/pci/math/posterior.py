from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def posterior_cov_gaussian_prior(
    gamma: torch.Tensor,
    sigma0: torch.Tensor,
) -> torch.Tensor:
    """
    Prior: x0 ~ N(0, Sigma0)
    Channel: Y = sqrt(gamma) x0 + eps, eps ~ N(0, I)
    Posterior covariance: (Sigma0^{-1} + gamma I)^{-1}
    gamma: (G,) tensor
    sigma0: (d,d) tensor
    returns: (G,d,d)
    """
    d = sigma0.shape[0]
    eye = torch.eye(d, device=sigma0.device, dtype=sigma0.dtype)
    sigma0_inv = torch.linalg.inv(sigma0)
    out = []
    for g in gamma:
        mat = sigma0_inv + g * eye
        out.append(torch.linalg.inv(mat))
    return torch.stack(out, dim=0)


@dataclass(frozen=True)
class DiscreteMixture:
    points: torch.Tensor  # (K, d)
    log_weights: torch.Tensor  # (K,)


def posterior_moments_discrete(
    y: torch.Tensor,  # (N,d)
    gamma: float,
    mix: DiscreteMixture,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Posterior over discrete support points with equal observation model.
    Returns (mean: (N,d), cov: (N,d,d)).
    """
    points = mix.points  # (K,d)
    log_pi = mix.log_weights  # (K,)
    sqrt_g = math.sqrt(gamma)
    # log p(y|xk) up to const: -0.5 || y - sqrt(g) xk ||^2
    diff = y[:, None, :] - sqrt_g * points[None, :, :]
    log_like = -0.5 * torch.sum(diff * diff, dim=-1)  # (N,K)
    log_post_unnorm = log_like + log_pi[None, :]
    log_post = log_post_unnorm - torch.logsumexp(log_post_unnorm, dim=-1, keepdim=True)
    w = torch.exp(log_post)  # (N,K)
    mean = w @ points  # (N,d)
    centered = points[None, :, :] - mean[:, None, :]  # (N,K,d)
    cov = torch.einsum("nk,nkd,nke->nde", w, centered, centered)
    return mean, cov


@dataclass(frozen=True)
class GaussianMixture:
    weights: torch.Tensor  # (K,)
    means: torch.Tensor  # (K,d)
    covs: torch.Tensor  # (K,d,d)


def posterior_moments_gmm(
    y: torch.Tensor,  # (N,d)
    gamma: float,
    gmm: GaussianMixture,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Exact posterior moments for Gaussian mixture prior under standardized channel.
    Returns (mean: (N,d), cov: (N,d,d)).
    """
    K, d = gmm.means.shape
    device, dtype = y.device, y.dtype
    eye = torch.eye(d, device=device, dtype=dtype)
    sqrt_g = math.sqrt(gamma)

    # For each component:
    # y marginal: y ~ N(sqrt_g * mu_k, I + gamma * Sigma_k)
    # posterior covariance in x-space: S_k = (Sigma_k^{-1} + gamma I)^{-1}
    # posterior mean: m_k = S_k (Sigma_k^{-1} mu_k + sqrt_g y)
    log_weights = torch.log(gmm.weights.to(device=device, dtype=dtype) + 1e-30)  # (K,)
    means = gmm.means.to(device=device, dtype=dtype)
    covs = gmm.covs.to(device=device, dtype=dtype)

    comp_loglik = []
    post_means = []
    post_covs = []
    for k in range(K):
        mu = means[k]
        Sigma = covs[k]
        Sigma_inv = torch.linalg.inv(Sigma)
        S = torch.linalg.inv(Sigma_inv + gamma * eye)
        post_covs.append(S)
        rhs = (Sigma_inv @ mu).unsqueeze(1) + sqrt_g * y.T  # (d,N)
        m = (S @ rhs).T  # (N,d)
        post_means.append(m)

        Vy = eye + gamma * Sigma
        # log N(y; sqrt_g mu, Vy)
        diff = y - sqrt_g * mu[None, :]
        Vy_inv = torch.linalg.inv(Vy)
        quad = torch.sum(diff * (diff @ Vy_inv), dim=1)
        sign, logdet = torch.linalg.slogdet(Vy)
        log_norm = -0.5 * (d * math.log(2 * math.pi) + logdet)
        comp_loglik.append(log_norm - 0.5 * quad)

    comp_loglik = torch.stack(comp_loglik, dim=1)  # (N,K)
    log_post_unnorm = comp_loglik + log_weights[None, :]
    log_post = log_post_unnorm - torch.logsumexp(log_post_unnorm, dim=1, keepdim=True)
    w = torch.exp(log_post)  # (N,K)

    post_means = torch.stack(post_means, dim=1)  # (N,K,d)
    mean = torch.einsum("nk,nkd->nd", w, post_means)

    # Cov = sum_k w_k (S_k + (m_k - mean)(m_k - mean)^T)
    cov = torch.zeros((y.shape[0], d, d), device=device, dtype=dtype)
    for k in range(K):
        mk = post_means[:, k, :]
        diffm = (mk - mean).unsqueeze(-1)  # (N,d,1)
        cov = cov + w[:, k].view(-1, 1, 1) * (post_covs[k].unsqueeze(0) + diffm @ diffm.transpose(1, 2))

    return mean, cov
