"""Sequential Monte Carlo for marginal likelihood / Bayes factor (Algorithm 1).

The marginal likelihood under outcome y is

    m_y(h) = int p(Delta x_{1:T} | v_{1:T}, y, theta) Pi(d theta).

We estimate it with the standard particle-filter unbiased estimator: at each
time t, accumulate
    log m_y += logsumexp_i ( log w_{t-1}^{(i)} + log f_y(Delta x_t | v_t, theta^{(i)}) ).
Particles are stored in unconstrained R^17 space; ESS-triggered systematic
resampling and a random-walk Metropolis rejuvenation kernel target the
filtering distribution Pi(theta | Delta x_{1:t}, v_{1:t}, y).

The Bayes-factor wrapper runs SMC twice (once per y) and returns the posterior
P(Y=1 | h) via Eq. 7.
"""

from __future__ import annotations

import math
from typing import Optional, Dict, Any

import numpy as np
from scipy.special import logsumexp

from . import model, priors


def _stable_sigmoid(x: float) -> float:
    """Sigmoid that avoids overflow for large |x|."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling. Returns indices of length N."""
    N = weights.shape[0]
    positions = (rng.uniform(0, 1) + np.arange(N)) / N
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0  # guard against rounding
    return np.searchsorted(cumsum, positions)


def _eval_per_step_logf(z: np.ndarray, dx_t: float, v_t: float, y: int) -> np.ndarray:
    """log f_y(dx_t | v_t, theta(z^{(i)})) for each particle i. Shape (N,)."""
    theta_batch, _ = priors.transform(z)
    dx_arr = np.array([dx_t])
    v_arr = np.array([v_t])
    return model.mixture_logpdf(dx_arr, v_arr, y, theta_batch)[..., 0]


def _eval_loglik_to_t(z: np.ndarray, dx: np.ndarray, v: np.ndarray, y: int,
                      t: int) -> np.ndarray:
    """sum_{s<=t} log f_y(dx_s | v_s, theta(z^{(i)})). Shape (N,)."""
    theta_batch, _ = priors.transform(z)
    return model.mixture_logpdf(dx[:t], v[:t], y, theta_batch).sum(axis=-1)


def _rwmh_rejuvenate(z: np.ndarray, dx: np.ndarray, v: np.ndarray, y: int, t: int,
                     n_steps: int, step_size: float, rng: np.random.Generator
                     ) -> tuple:
    """Random-walk Metropolis sweeps targeting Pi(theta | Delta x_{1:t}, v_{1:t}, y).

    Returns (z_new, acceptance_rate).
    """
    N, d = z.shape
    log_prior_z = priors.log_prior_unconstrained(z)
    log_lik_z = _eval_loglik_to_t(z, dx, v, y, t)
    log_target = log_prior_z + log_lik_z

    accepts = 0
    total = 0
    for _ in range(n_steps):
        z_prop = z + step_size * rng.standard_normal(size=z.shape)
        log_prior_p = priors.log_prior_unconstrained(z_prop)
        log_lik_p = _eval_loglik_to_t(z_prop, dx, v, y, t)
        log_target_p = log_prior_p + log_lik_p

        log_alpha = log_target_p - log_target
        u = rng.uniform(size=N)
        accept = np.log(u) < log_alpha

        z = np.where(accept[:, None], z_prop, z)
        log_target = np.where(accept, log_target_p, log_target)
        accepts += int(accept.sum())
        total += N

    return z, (accepts / max(total, 1))


def smc_marginal_likelihood(
    dx: np.ndarray,
    v: np.ndarray,
    y: int,
    n_particles: int = 1000,
    ess_threshold: float = 0.5,
    mcmc_steps: int = 5,
    initial_step_size: float = 0.3,
    record_pi_t: bool = False,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run Algorithm 1 to estimate log m_y(h) for a single outcome y.

    Parameters
    ----------
    dx, v : (T,) arrays of log-odds increments and traded volumes.
    y : 0 or 1.
    n_particles : N in Algorithm 1.
    ess_threshold : resample/rejuvenate when ESS < ess_threshold * N.
    mcmc_steps : number of RW-Metropolis sweeps per rejuvenation.
    initial_step_size : starting RW-Metropolis std in unconstrained space (adapted
        toward a target acceptance rate of 0.234).
    record_pi_t : if True, also return per-step incremental log-evidence so that
        the caller can build an online posterior trace.

    Returns
    -------
    {
        "log_my":      float estimate of log m_y(h),
        "ess_history": (T,) effective sample size each step,
        "particles_z": (N, 17) final particle positions (unconstrained),
        "weights":     (N,) final normalized weights,
        "log_inc":     (T,) per-step log-increments to log m_y if record_pi_t,
    }
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    T = dx.shape[0]
    N = n_particles
    d = priors.UNCONSTRAINED_DIM

    # Initialize particles by sampling from the prior, then mapping to z.
    theta_init = priors.sample_prior_batched(rng, N)
    z = priors.to_unconstrained(theta_init)

    log_w = np.full(N, -math.log(N))  # log normalized weights
    log_my = 0.0

    ess_history = np.empty(T)
    log_inc = np.empty(T) if record_pi_t else None
    step_size = initial_step_size

    for t in range(T):
        log_f = _eval_per_step_logf(z, dx[t], v[t], y)        # (N,)
        log_w_unnorm = log_w + log_f
        # Marginal-likelihood increment: logsumexp over particles
        inc = logsumexp(log_w_unnorm)                          # log sum_i w_{t-1}^i f_t^i
        log_my += inc
        if record_pi_t:
            log_inc[t] = inc

        # Normalize weights
        log_w = log_w_unnorm - inc
        w = np.exp(log_w)
        w = w / w.sum()  # numerical safety
        ess = 1.0 / np.sum(w * w)
        ess_history[t] = ess

        if ess < ess_threshold * N and t < T - 1:
            # Resample
            idx = _systematic_resample(w, rng)
            z = z[idx]
            log_w = np.full(N, -math.log(N))

            # Rejuvenate via RW-Metropolis targeting Pi(theta | data so far)
            z, accept_rate = _rwmh_rejuvenate(
                z, dx, v, y, t + 1, mcmc_steps, step_size, rng
            )
            # Adapt step size toward 0.234 acceptance
            if accept_rate > 0.30:
                step_size *= 1.2
            elif accept_rate < 0.15:
                step_size *= 0.8
            if verbose:
                print(f"  [smc y={y}] t={t} resample+rejuv  acc={accept_rate:.2f}  step={step_size:.3f}")

    out = {
        "log_my": float(log_my),
        "ess_history": ess_history,
        "particles_z": z,
        "weights": np.exp(log_w),
    }
    if record_pi_t:
        out["log_inc"] = log_inc
    return out


def bayes_factor_smc(
    dx: np.ndarray,
    v: np.ndarray,
    pi0: float = 0.5,
    n_particles: int = 1000,
    ess_threshold: float = 0.5,
    mcmc_steps: int = 5,
    seed: Optional[int] = None,
    record_pi_t: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run SMC for both outcomes; return Bayes factor and posterior on Y.

    By Eq. 7, log posterior odds = logit(pi0) + log BF_T(h).
    """
    rng = np.random.default_rng(seed)
    # Use distinct sub-rngs so the two SMC runs don't share random numbers.
    seed0 = int(rng.integers(0, 2**31 - 1))
    seed1 = int(rng.integers(0, 2**31 - 1))

    res0 = smc_marginal_likelihood(
        dx, v, y=0, n_particles=n_particles, ess_threshold=ess_threshold,
        mcmc_steps=mcmc_steps, record_pi_t=record_pi_t, seed=seed0, verbose=verbose,
    )
    res1 = smc_marginal_likelihood(
        dx, v, y=1, n_particles=n_particles, ess_threshold=ess_threshold,
        mcmc_steps=mcmc_steps, record_pi_t=record_pi_t, seed=seed1, verbose=verbose,
    )

    log_BF = res1["log_my"] - res0["log_my"]
    logit_pi0 = math.log(pi0 / (1.0 - pi0))
    log_post_odds = logit_pi0 + log_BF
    posterior = _stable_sigmoid(log_post_odds)

    out = {
        "log_BF": float(log_BF),
        "log_m0": float(res0["log_my"]),
        "log_m1": float(res1["log_my"]),
        "posterior": float(posterior),
        "pi0": float(pi0),
        "smc0": res0,
        "smc1": res1,
    }
    if record_pi_t:
        # Online posterior pi_t = sigma(logit pi0 + cumsum(log_inc^1) - cumsum(log_inc^0))
        cum1 = np.cumsum(res1["log_inc"])
        cum0 = np.cumsum(res0["log_inc"])
        log_post_odds_t = logit_pi0 + cum1 - cum0
        # Numerically-stable sigmoid (avoids overflow when |log_odds| is large)
        out["pi_t"] = np.where(
            log_post_odds_t >= 0,
            1.0 / (1.0 + np.exp(-np.clip(log_post_odds_t, -500, 500))),
            np.exp(np.clip(log_post_odds_t, -500, 500)) /
                (1.0 + np.exp(np.clip(log_post_odds_t, -500, 500))),
        )
        out["log_BF_t"] = cum1 - cum0
    return out
