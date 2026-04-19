"""Sequential Monte Carlo for the volume-extended model.

This mirrors `smc.py` but targets the joint model

    p(dx_{1:T}, v_{1:T} | y, theta_x, theta_v)

where:
- `theta_x` are the original latent-type parameters from `priors.py`/`model.py`
- `theta_v` are volume Markov parameters from `priors_vol.py`/`model_vol.py`

The core SMC algorithm is unchanged: sequential weighting, ESS-triggered
systematic resampling, and random-walk Metropolis rejuvenation in unconstrained
space.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.special import logsumexp

from . import model, model_vol, priors, priors_vol


def _stable_sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    N = weights.shape[0]
    positions = (rng.uniform(0, 1) + np.arange(N)) / N
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    return np.searchsorted(cumsum, positions)


def _dims() -> Tuple[int, int, int]:
    d_x = priors.UNCONSTRAINED_DIM
    d_v = priors_vol.UNCONSTRAINED_DIM_VOL
    return d_x, d_v, d_x + d_v


def _split_z(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    d_x, _, _ = _dims()
    return z[..., :d_x], z[..., d_x:]


def _transform_both(z: np.ndarray):
    z_x, z_v = _split_z(z)
    theta_x, _ = priors.transform(z_x)
    theta_v, _ = priors_vol.transform(z_v)
    return theta_x, theta_v


def _sample_joint_prior_unconstrained(
    rng: np.random.Generator,
    n: int,
) -> np.ndarray:
    theta_x = priors.sample_prior_batched(rng, n)
    theta_v = priors_vol.sample_prior_batched(rng, n)

    z_x = priors.to_unconstrained(theta_x)
    z_v = priors_vol.to_unconstrained(theta_v)
    return np.concatenate([z_x, z_v], axis=-1)


def _eval_per_step_logf(z: np.ndarray, dx: np.ndarray, v: np.ndarray, y: int, t: int) -> np.ndarray:
    """Per-step joint log factor for particle batch at time t.

    For t = 0:
      log f_0 = log p(dx_0 | v_0, y, theta_x)
      (volume initial term omitted as constant wrt theta by design)

    For t >= 1:
      log f_t = log p(dx_t | v_t, y, theta_x) + log p(v_t | v_{t-1}, theta_v)
    """
    theta_x, theta_v = _transform_both(z)

    dx_t_arr = np.array([dx[t]], dtype=np.float64)
    v_t_arr = np.array([v[t]], dtype=np.float64)
    log_dx_t = model.mixture_logpdf(dx_t_arr, v_t_arr, y, theta_x)[..., 0]

    if t == 0:
        return log_dx_t

    # Extract only the transition contribution at index 1 from [v_{t-1}, v_t].
    v_pair = np.array([v[t - 1], v[t]], dtype=np.float64)
    log_v_pair = model_vol.volume_transition_logpdf(
        v_pair,
        theta_v,
        include_initial=False,
    )
    log_v_t = log_v_pair[..., 1]
    return log_dx_t + log_v_t


def _eval_loglik_to_t(z: np.ndarray, dx: np.ndarray, v: np.ndarray, y: int, t: int) -> np.ndarray:
    """Joint log-likelihood sum_{s<=t} for each particle.

    `t` is a length (exclusive upper bound), matching the convention in `smc.py`.
    """
    theta_x, theta_v = _transform_both(z)
    return model_vol.joint_loglik(
        dx[:t],
        v[:t],
        y,
        theta_x,
        theta_v,
        include_initial=False,
    )


def _log_prior_unconstrained_joint(z: np.ndarray) -> np.ndarray:
    z_x, z_v = _split_z(z)
    return priors.log_prior_unconstrained(z_x) + priors_vol.log_prior_unconstrained(z_v)


def _rwmh_rejuvenate(
    z: np.ndarray,
    dx: np.ndarray,
    v: np.ndarray,
    y: int,
    t: int,
    n_steps: int,
    step_size: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    N, _ = z.shape

    log_prior_z = _log_prior_unconstrained_joint(z)
    log_lik_z = _eval_loglik_to_t(z, dx, v, y, t)
    log_target = log_prior_z + log_lik_z

    accepts = 0
    total = 0

    for _ in range(n_steps):
        z_prop = z + step_size * rng.standard_normal(size=z.shape)
        log_prior_p = _log_prior_unconstrained_joint(z_prop)
        log_lik_p = _eval_loglik_to_t(z_prop, dx, v, y, t)
        log_target_p = log_prior_p + log_lik_p

        log_alpha = log_target_p - log_target
        accept = np.log(rng.uniform(size=N)) < log_alpha

        z = np.where(accept[:, None], z_prop, z)
        log_target = np.where(accept, log_target_p, log_target)
        accepts += int(accept.sum())
        total += N

    return z, (accepts / max(total, 1))


def smc_marginal_likelihood_vol(
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
    """SMC marginal likelihood for the volume-extended model under fixed y."""
    if rng is None:
        rng = np.random.default_rng(seed)

    T = dx.shape[0]
    N = n_particles
    _, _, d = _dims()

    z = _sample_joint_prior_unconstrained(rng, N)

    log_w = np.full(N, -math.log(N))
    log_my = 0.0

    ess_history = np.empty(T)
    log_inc = np.empty(T) if record_pi_t else None
    step_size = initial_step_size

    for t in range(T):
        log_f = _eval_per_step_logf(z, dx, v, y, t)
        log_w_unnorm = log_w + log_f

        inc = logsumexp(log_w_unnorm)
        log_my += inc
        if record_pi_t:
            log_inc[t] = inc

        log_w = log_w_unnorm - inc
        w = np.exp(log_w)
        w = w / w.sum()
        ess = 1.0 / np.sum(w * w)
        ess_history[t] = ess

        if ess < ess_threshold * N and t < T - 1:
            idx = _systematic_resample(w, rng)
            z = z[idx]
            log_w = np.full(N, -math.log(N))

            z, accept_rate = _rwmh_rejuvenate(
                z,
                dx,
                v,
                y,
                t + 1,
                mcmc_steps,
                step_size,
                rng,
            )

            if accept_rate > 0.30:
                step_size *= 1.2
            elif accept_rate < 0.15:
                step_size *= 0.8

            if verbose:
                print(
                    f"  [smc_vol y={y}] t={t} resample+rejuv  "
                    f"acc={accept_rate:.2f}  step={step_size:.3f}"
                )

    out: Dict[str, Any] = {
        "log_my": float(log_my),
        "ess_history": ess_history,
        "particles_z": z,
        "weights": np.exp(log_w),
    }
    if record_pi_t:
        out["log_inc"] = log_inc
    return out


def bayes_factor_smc_vol(
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
    """Run volume-extended SMC under y=0,1 and return BF and posterior."""
    rng = np.random.default_rng(seed)
    seed0 = int(rng.integers(0, 2**31 - 1))
    seed1 = int(rng.integers(0, 2**31 - 1))

    res0 = smc_marginal_likelihood_vol(
        dx,
        v,
        y=0,
        n_particles=n_particles,
        ess_threshold=ess_threshold,
        mcmc_steps=mcmc_steps,
        record_pi_t=record_pi_t,
        seed=seed0,
        verbose=verbose,
    )
    res1 = smc_marginal_likelihood_vol(
        dx,
        v,
        y=1,
        n_particles=n_particles,
        ess_threshold=ess_threshold,
        mcmc_steps=mcmc_steps,
        record_pi_t=record_pi_t,
        seed=seed1,
        verbose=verbose,
    )

    log_BF = res1["log_my"] - res0["log_my"]
    logit_pi0 = math.log(pi0 / (1.0 - pi0))
    log_post_odds = logit_pi0 + log_BF
    posterior = _stable_sigmoid(log_post_odds)

    out: Dict[str, Any] = {
        "log_BF": float(log_BF),
        "log_m0": float(res0["log_my"]),
        "log_m1": float(res1["log_my"]),
        "posterior": float(posterior),
        "pi0": float(pi0),
        "smc0": res0,
        "smc1": res1,
    }

    if record_pi_t:
        cum1 = np.cumsum(res1["log_inc"])
        cum0 = np.cumsum(res0["log_inc"])
        log_post_odds_t = logit_pi0 + cum1 - cum0
        out["pi_t"] = np.where(
            log_post_odds_t >= 0,
            1.0 / (1.0 + np.exp(-np.clip(log_post_odds_t, -500, 500))),
            np.exp(np.clip(log_post_odds_t, -500, 500))
            / (1.0 + np.exp(np.clip(log_post_odds_t, -500, 500))),
        )
        out["log_BF_t"] = cum1 - cum0

    return out
