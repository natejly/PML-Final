"""Mean-field reparameterized variational inference for the Bayes factor (Algorithm 2).

We approximate the posterior over the unconstrained parameter z in R^17 with a
diagonal Gaussian q(z; phi) = N(mu_phi, diag(sigma_phi^2)). The ELBO under
outcome y is

    L(phi; y) = E_{z ~ q} [log p(Delta x | v, y, theta(z)) + log pi(z)] + H[q],

where pi(z) = log Pi(theta(z)) + log|det dtheta/dz| is the prior in
unconstrained space and H[q] is the differential entropy of q. We optimize phi
by stochastic gradient ascent (Adam) with the reparameterization trick.

By Eq. 7 the (approximate) log Bayes factor is L^*(1) - L^*(0). Per Remark 5.1
this is biased (the difference of two ELBOs is not bounded by log BF in either
direction), so callers should treat the resulting posterior as approximate.
"""

from __future__ import annotations

import math
from typing import Optional, Dict, Any

import numpy as np
import torch

from . import model, priors


def _gaussian_entropy(log_sigma: torch.Tensor) -> torch.Tensor:
    """Entropy of N(mu, diag(sigma^2)). Returns scalar.

    H = sum_d (0.5 * log(2*pi*e) + log sigma_d).
    """
    d = log_sigma.shape[-1]
    return 0.5 * d * (1.0 + math.log(2.0 * math.pi)) + log_sigma.sum()


def _elbo_estimate(mu: torch.Tensor, log_sigma: torch.Tensor,
                   dx_t: torch.Tensor, v_t: torch.Tensor, y: int,
                   n_samples: int) -> torch.Tensor:
    """Single-batch reparameterized estimate of L(phi; y)."""
    sigma = log_sigma.exp()
    eps = torch.randn(n_samples, mu.shape[0], dtype=mu.dtype, device=mu.device)
    z = mu + sigma * eps                                    # (S, d)

    log_lik = model.loglik(dx_t, v_t, y, priors.transform(z)[0])  # (S,)
    log_prior_z = priors.log_prior_unconstrained(z)               # (S,)
    expected_joint = (log_lik + log_prior_z).mean()

    return expected_joint + _gaussian_entropy(log_sigma)


def vi_marginal_likelihood(
    dx: np.ndarray,
    v: np.ndarray,
    y: int,
    n_steps: int = 1500,
    n_samples: int = 8,
    learning_rate: float = 0.05,
    init_log_sigma: float = -1.5,
    seed: Optional[int] = None,
    progress: bool = False,
) -> Dict[str, Any]:
    """Run mean-field VI to estimate L^*(y) and return the optimized variational
    parameters.

    Returns a dict with `elbo`, `mu`, `log_sigma`, `theta_mean` (variational
    posterior mean of theta after the bijector), `elbo_trace`.
    """
    if seed is not None:
        torch.manual_seed(seed)

    d = priors.UNCONSTRAINED_DIM
    dtype = torch.float64
    device = torch.device("cpu")

    # Initialize: small means biased to soft prior centers, modest scale.
    mu = torch.zeros(d, dtype=dtype, device=device, requires_grad=True)
    log_sigma = torch.full((d,), init_log_sigma, dtype=dtype, device=device,
                           requires_grad=True)

    dx_t = torch.as_tensor(dx, dtype=dtype, device=device)
    v_t = torch.as_tensor(v, dtype=dtype, device=device)

    optimizer = torch.optim.Adam([mu, log_sigma], lr=learning_rate)

    elbo_trace = np.empty(n_steps)
    iterator = range(n_steps)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=f"VI y={y}")
        except ImportError:
            pass

    for step in iterator:
        optimizer.zero_grad()
        elbo = _elbo_estimate(mu, log_sigma, dx_t, v_t, y, n_samples)
        loss = -elbo
        loss.backward()
        torch.nn.utils.clip_grad_norm_([mu, log_sigma], max_norm=10.0)
        optimizer.step()
        elbo_trace[step] = elbo.detach().item()

    # Final ELBO with more samples for a less noisy estimate.
    with torch.no_grad():
        elbo_final = _elbo_estimate(mu, log_sigma, dx_t, v_t, y, n_samples=128)

    # Variational posterior mean of theta (push mu through the bijector).
    with torch.no_grad():
        theta_mean_t, _ = priors.transform(mu)
        theta_mean = {k: v_.detach().cpu().numpy() for k, v_ in theta_mean_t.items()}

    return {
        "elbo": float(elbo_final.item()),
        "mu": mu.detach().cpu().numpy(),
        "log_sigma": log_sigma.detach().cpu().numpy(),
        "theta_mean": theta_mean,
        "elbo_trace": elbo_trace,
    }


def bayes_factor_vi(
    dx: np.ndarray,
    v: np.ndarray,
    pi0: float = 0.5,
    n_steps: int = 1500,
    n_samples: int = 8,
    learning_rate: float = 0.05,
    seed: Optional[int] = None,
    progress: bool = False,
) -> Dict[str, Any]:
    """Run VI under both outcomes; approximate log BF as L^*(1) - L^*(0).

    By Remark 5.1 this estimator is biased; report it as an approximation.
    """
    res0 = vi_marginal_likelihood(
        dx, v, y=0, n_steps=n_steps, n_samples=n_samples,
        learning_rate=learning_rate, seed=seed, progress=progress,
    )
    res1 = vi_marginal_likelihood(
        dx, v, y=1, n_steps=n_steps, n_samples=n_samples,
        learning_rate=learning_rate,
        seed=None if seed is None else seed + 1, progress=progress,
    )

    from .smc import _stable_sigmoid
    log_BF = res1["elbo"] - res0["elbo"]
    logit_pi0 = math.log(pi0 / (1.0 - pi0))
    log_post_odds = logit_pi0 + log_BF
    posterior = _stable_sigmoid(log_post_odds)

    return {
        "log_BF": float(log_BF),
        "elbo_y0": float(res0["elbo"]),
        "elbo_y1": float(res1["elbo"]),
        "posterior": float(posterior),
        "pi0": float(pi0),
        "vi0": res0,
        "vi1": res1,
    }
