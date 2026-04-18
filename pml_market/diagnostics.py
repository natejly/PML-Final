"""Identifiability, information-gain and stability diagnostics.

Implements:
  - kl_projection_gap : MC + SGD estimate of delta_T(y_star, theta_star) (Def. 4.1).
  - effective_informativeness : eta(v; theta) (Def. 4.3).
  - realized_information_gain : IG(h) = KL(Bern(pi_T) || Bern(pi_0)) (Eq. 19).
  - perturb_history : Gaussian perturbation of Delta x for Experiment 3.
  - stability_bound : RHS of Theorem 4.4 (linear-in-perturbation Lipschitz bound).
"""

from __future__ import annotations

import math
from typing import Dict, Any, Optional

import numpy as np
import torch

from . import synthetic
from .core import Model, Prior
from .model import GaussianLatentTypeModel, PARAM_NAMES as _DEFAULT_PARAM_NAMES
from .priors import LatentTypePrior


# ---------------------------------------------------------------------------
# Realized information gain (Eq. 19)
# ---------------------------------------------------------------------------

def realized_information_gain(pi_T: float, pi0: float = 0.5,
                              eps: float = 1e-12) -> float:
    """KL(Bern(pi_T) || Bern(pi0))."""
    p, q = float(pi_T), float(pi0)
    p = min(max(p, eps), 1.0 - eps)
    q = min(max(q, eps), 1.0 - eps)
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def information_gain_trace(pi_t: np.ndarray, pi0: float = 0.5) -> np.ndarray:
    """IG(H_t) for each t. Used for Experiment 4 plots."""
    pi_t = np.clip(pi_t, 1e-12, 1.0 - 1e-12)
    q = float(pi0)
    return pi_t * np.log(pi_t / q) + (1.0 - pi_t) * np.log((1.0 - pi_t) / (1.0 - q))


# ---------------------------------------------------------------------------
# Effective informativeness (Def. 4.3)
# ---------------------------------------------------------------------------

def effective_informativeness(v: np.ndarray, theta: Dict[str, np.ndarray]
                              ) -> np.ndarray:
    """eta(v_t; theta) = sum_k rho_k(v_t; theta) * sign(m_{k,1} - m_{k,0}) * |m_{k,1} - m_{k,0}|.

    For the Gaussian model in Example 3.1:
      Type 1 (informed): m_{1,1} - m_{1,0} = +2 * mu1 * (1 - exp(-lam1 v))
      Type 2 (noise):    0
      Type 3 (manipulator): m_{3,1} - m_{3,0} = -2 * mu3 * 1{v > tau3}

    Returns shape (T,).
    """
    v = np.asarray(v)
    omega = np.asarray(theta["omega"])
    gamma = np.asarray(theta["gamma"])

    # Gating
    log_omega = np.log(omega)
    g0 = gamma[:, 0]
    g1 = gamma[:, 1]
    log1pv = np.log1p(v)[:, None]
    logits = log_omega[None, :] + g0[None, :] + g1[None, :] * log1pv
    logits -= logits.max(axis=-1, keepdims=True)
    rho = np.exp(logits)
    rho /= rho.sum(axis=-1, keepdims=True)  # (T, K)

    diff1 = 2.0 * float(theta["mu1"]) * (1.0 - np.exp(-float(theta["lam1"]) * v))
    diff2 = np.zeros_like(v)
    active = (v > float(theta["tau3"])).astype(float)
    diff3 = -2.0 * float(theta["mu3"]) * active

    diffs = np.stack([diff1, diff2, diff3], axis=-1)  # (T, K)
    eta = (rho * np.sign(diffs) * np.abs(diffs)).sum(axis=-1)
    return eta


# ---------------------------------------------------------------------------
# KL projection gap (Def. 4.1) via Monte Carlo + Adam
# ---------------------------------------------------------------------------

def _theta_dict_to_torch(theta: Dict[str, np.ndarray],
                         param_names=_DEFAULT_PARAM_NAMES,
                         ) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(np.asarray(theta[k]), dtype=torch.float64)
            for k in param_names}


def kl_projection_gap(
    theta_star: Dict[str, np.ndarray],
    v: np.ndarray,
    y_star: int = 1,
    n_iter: int = 600,
    n_samples: int = 256,
    learning_rate: float = 0.05,
    seed: Optional[int] = None,
    model: Optional[Model] = None,
    prior: Optional[Prior] = None,
) -> Dict[str, Any]:
    """Estimate delta_T(y_star, theta_star) = inf_theta (1/T) D_KL(P_{y*,theta*} || P_{1-y*,theta}).

    We minimize a Monte Carlo estimate of the per-step KL by drawing
    Delta x ~ P_{y*,theta*} and pushing theta around by Adam in the
    unconstrained space.
    """
    if model is None:
        model = GaussianLatentTypeModel()
    if prior is None:
        prior = LatentTypePrior()

    if seed is not None:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    T = v.shape[0]
    y_alt = 1 - y_star

    # Pre-sample Delta x ~ P_{y*, theta*} once (n_samples per time step).
    dx_samples = np.empty((n_samples, T))
    for s in range(n_samples):
        dx_samples[s] = synthetic.simulate_increments(v, y_star, theta_star, rng)
    dx_t = torch.as_tensor(dx_samples, dtype=torch.float64)
    v_t = torch.as_tensor(v, dtype=torch.float64)

    # log p_{y*, theta*}(dx) needed for the KL definition; constant in theta.
    theta_star_t = _theta_dict_to_torch(theta_star, model.PARAM_NAMES)
    log_p_star = torch.stack([
        model.mixture_logpdf(dx_t[s], v_t, y_star, theta_star_t)
        for s in range(n_samples)
    ], dim=0)  # (S, T)

    # Optimise theta in unconstrained space.
    z = torch.zeros(prior.UNCONSTRAINED_DIM, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([z], lr=learning_rate)

    trace = np.empty(n_iter)
    for step in range(n_iter):
        opt.zero_grad()
        theta_alt, _ = prior.transform(z)
        log_p_alt = torch.stack([
            model.mixture_logpdf(dx_t[s], v_t, y_alt, theta_alt)
            for s in range(n_samples)
        ], dim=0)  # (S, T)
        per_step_kl = (log_p_star - log_p_alt).mean(dim=0)  # (T,)
        kl_per_step = per_step_kl.mean()                    # scalar = (1/T) sum kl_t
        loss = kl_per_step
        loss.backward()
        torch.nn.utils.clip_grad_norm_([z], max_norm=10.0)
        opt.step()
        trace[step] = float(loss.item())

    delta_T = float(trace[-50:].mean())
    with torch.no_grad():
        theta_proj_t, _ = prior.transform(z)
        theta_proj = {k: v_.detach().cpu().numpy() for k, v_ in theta_proj_t.items()}

    return {
        "delta_T": delta_T,
        "theta_proj": theta_proj,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Perturbation utilities (Section 4.5 / Experiment 3)
# ---------------------------------------------------------------------------

def perturb_history(dx: np.ndarray, sigma: float,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Add iid N(0, sigma^2) noise to the increments for Experiment 3."""
    if rng is None:
        rng = np.random.default_rng()
    return dx + sigma * rng.standard_normal(size=dx.shape)


def stability_bound(dx: np.ndarray, dx_perturbed: np.ndarray,
                    v: np.ndarray, v_perturbed: Optional[np.ndarray] = None,
                    Lx: float = 4.0, Lv: float = 0.0) -> float:
    """RHS of Theorem 4.4 (Eq. 17):
        |log BF(h) - log BF(h')| <= 2 (Lx sum |dx - dx'| + Lv sum |v - v'|).

    For the Gaussian latent-type model on |dx| <= R, Lx scales as c0 + c1 * R.
    Default Lx=4 is a moderate choice; callers should tune to the empirical
    truncation radius R (e.g. 0.99 quantile of |dx|).
    """
    dx_term = np.sum(np.abs(dx - dx_perturbed))
    v_term = 0.0 if v_perturbed is None else np.sum(np.abs(v - v_perturbed))
    return 2.0 * (Lx * dx_term + Lv * v_term)


def gaussian_lipschitz_constant(R: float, sigma_min: float = 0.1) -> float:
    """A conservative Lx(R) for the Gaussian latent-type model.

    For a Gaussian density with scale s, |d log f / dx| = |x - mean| / s^2
    grows linearly in |x|. With |x| <= R and s >= sigma_min,
        Lx(R) <= (R + |mean_max|) / sigma_min^2.
    We use a simple proxy Lx(R) = R / sigma_min^2 (the location |mean_max| is
    small compared to R for typical R).
    """
    return R / (sigma_min ** 2)


# ---------------------------------------------------------------------------
# Online posterior helper (re-export from SMC for convenience)
# ---------------------------------------------------------------------------

def online_posterior(dx: np.ndarray, v: np.ndarray, pi0: float = 0.5,
                     n_particles: int = 1000, mcmc_steps: int = 5,
                     seed: Optional[int] = None,
                     model: Optional[Model] = None,
                     prior: Optional[Prior] = None) -> Dict[str, Any]:
    """Run SMC under both outcomes with ``record_pi_t=True`` and return the
    online posterior trace pi_t along with the final summary.

    Defaults to the Gaussian latent-type model + matching prior; pass
    alternative `Model` / `Prior` instances to swap them in.
    """
    from .smc import SMCInference
    if model is None:
        model = GaussianLatentTypeModel()
    if prior is None:
        prior = LatentTypePrior()
    smc = SMCInference(n_particles=n_particles, mcmc_steps=mcmc_steps)
    return smc.run(dx, v, model, prior, pi0=pi0, seed=seed, record_pi_t=True)
