"""Sequential Monte Carlo for marginal likelihood / Bayes factor (Algorithm 1).

The marginal likelihood under outcome y is

    m_y(h) = int p(obs_{1:T} | y, theta) Pi(d theta).

We estimate it with the standard particle-filter unbiased estimator: at each
time t, accumulate
    log m_y += logsumexp_i (
        log w_{t-1}^{(i)}
        + log p(obs_t | obs_{<t}, y, theta^{(i)})
    ).
Particles are stored in the prior's unconstrained space; ESS-triggered
systematic resampling and a random-walk Metropolis rejuvenation kernel target
the filtering distribution Pi(theta | Delta x_{1:t}, v_{1:t}, y).

The `SMCInference` class runs SMC twice (once per y) and returns the posterior
P(Y=1 | h) via Eq. 7.  Both the model and the prior are injected, so the same
SMC engine works on any (Model, Prior) subclass pair.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
from scipy.special import logsumexp

from ..core import Inference, Model, Prior


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
    cumsum[-1] = 1.0
    return np.searchsorted(cumsum, positions)


# ---------------------------------------------------------------------------
# Per-particle likelihood evaluators (private, take Model + Prior)
# ---------------------------------------------------------------------------


def _eval_per_step_logf(z: np.ndarray, dx: np.ndarray, v: np.ndarray, y: int, t: int,
                        model: Model, prior: Prior) -> np.ndarray:
    """One-step log factor for each particle. Shape (N,)."""
    theta_batch, _ = prior.transform(z)
    return model.incremental_logpdf(dx, v, y, theta_batch, t)


def _eval_loglik_to_t(z: np.ndarray, dx: np.ndarray, v: np.ndarray, y: int,
                      t: int, model: Model, prior: Prior) -> np.ndarray:
    """Prefix log-likelihood for each particle. Shape (N,)."""
    theta_batch, _ = prior.transform(z)
    return model.loglik(dx[:t], v[:t], y, theta_batch)


def _rwmh_rejuvenate(z: np.ndarray, dx: np.ndarray, v: np.ndarray, y: int, t: int,
                     n_steps: int, step_size: float,
                     model: Model, prior: Prior,
                     rng: np.random.Generator) -> tuple:
    """Random-walk Metropolis sweeps targeting Pi(theta | h_{1:t}, y).

    Returns (z_new, acceptance_rate).
    """
    N = z.shape[0]
    log_prior_z = prior.log_prior_unconstrained(z)
    log_lik_z = _eval_loglik_to_t(z, dx, v, y, t, model, prior)
    log_target = log_prior_z + log_lik_z

    accepts = 0
    total = 0
    for _ in range(n_steps):
        z_prop = z + step_size * rng.standard_normal(size=z.shape)
        log_prior_p = prior.log_prior_unconstrained(z_prop)
        log_lik_p = _eval_loglik_to_t(z_prop, dx, v, y, t, model, prior)
        log_target_p = log_prior_p + log_lik_p

        log_alpha = log_target_p - log_target
        u = rng.uniform(size=N)
        accept = np.log(u) < log_alpha

        z = np.where(accept[:, None], z_prop, z)
        log_target = np.where(accept, log_target_p, log_target)
        accepts += int(accept.sum())
        total += N

    return z, (accepts / max(total, 1))


# ---------------------------------------------------------------------------
# SMCInference class (public API)
# ---------------------------------------------------------------------------


class SMCInference(Inference):
    """Algorithm 1: SMC marginal-likelihood estimator + Bayes factor wrapper.

    Hyperparameters live on the instance; algorithm-agnostic arguments such as
    the prior odds ``pi0`` and the random ``seed`` are passed at call time.

    Parameters
    ----------
    n_particles : int
        Number of particles N.
    ess_threshold : float
        Resample/rejuvenate when ESS < ess_threshold * N.
    mcmc_steps : int
        Number of RW-Metropolis sweeps per rejuvenation.
    initial_step_size : float
        Starting RW-MH std in unconstrained space.  Adapted multiplicatively
        toward an acceptance rate inside [accept_shrink_below,
        accept_grow_above].
    accept_grow_above : float
        Multiply step_size by 1.2 if the empirical acceptance rate exceeds
        this threshold.  Default 0.30 (matches the pre-OOP implementation).
    accept_shrink_below : float
        Multiply step_size by 0.8 if the empirical acceptance rate falls
        below this threshold.  Default 0.15 (matches the pre-OOP
        implementation).
    verbose : bool
        Print per-step rejuvenation diagnostics.
    """

    def __init__(
        self,
        n_particles: int = 1000,
        ess_threshold: float = 0.5,
        mcmc_steps: int = 5,
        initial_step_size: float = 0.3,
        accept_grow_above: float = 0.30,
        accept_shrink_below: float = 0.15,
        verbose: bool = False,
    ):
        self.n_particles = int(n_particles)
        self.ess_threshold = float(ess_threshold)
        self.mcmc_steps = int(mcmc_steps)
        self.initial_step_size = float(initial_step_size)
        self.accept_grow_above = float(accept_grow_above)
        self.accept_shrink_below = float(accept_shrink_below)
        self.verbose = bool(verbose)

    def __repr__(self) -> str:
        return (f"SMCInference(n_particles={self.n_particles}, "
                f"ess_threshold={self.ess_threshold}, mcmc_steps={self.mcmc_steps})")

    # ------------------------------------------------------------------
    # Single-outcome marginal likelihood (Algorithm 1)
    # ------------------------------------------------------------------

    def marginal_likelihood(
        self,
        dx: np.ndarray,
        v: np.ndarray,
        y: int,
        model: Model,
        prior: Prior,
        *,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        record_pi_t: bool = False,
    ) -> Dict[str, Any]:
        """Estimate log m_y(h) for a single outcome y.

        Returns
        -------
        dict
            ``log_my``, ``ess_history``, ``particles_z``, ``weights``;
            plus ``log_inc`` (T,) when ``record_pi_t=True``.
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        T = dx.shape[0]
        N = self.n_particles

        theta_init = prior.sample(rng, N)
        z = prior.to_unconstrained(theta_init)

        log_w = np.full(N, -math.log(N))
        log_my = 0.0

        ess_history = np.empty(T)
        log_inc = np.empty(T) if record_pi_t else None
        step_size = self.initial_step_size

        for t in range(T):
            log_f = _eval_per_step_logf(z, dx, v, y, t, model, prior)
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

            if ess < self.ess_threshold * N and t < T - 1:
                idx = _systematic_resample(w, rng)
                z = z[idx]
                log_w = np.full(N, -math.log(N))

                z, accept_rate = _rwmh_rejuvenate(
                    z, dx, v, y, t + 1, self.mcmc_steps, step_size,
                    model, prior, rng,
                )
                if accept_rate > self.accept_grow_above:
                    step_size *= 1.2
                elif accept_rate < self.accept_shrink_below:
                    step_size *= 0.8
                if self.verbose:
                    print(f"  [smc y={y}] t={t} resample+rejuv  "
                          f"acc={accept_rate:.2f}  step={step_size:.3f}")

        out = {
            "log_my": float(log_my),
            "ess_history": ess_history,
            "particles_z": z,
            "weights": np.exp(log_w),
        }
        if record_pi_t:
            out["log_inc"] = log_inc
        return out

    # ------------------------------------------------------------------
    # Bayes factor / posterior (Inference interface)
    # ------------------------------------------------------------------

    def run(
        self,
        dx: np.ndarray,
        v: np.ndarray,
        model: Model,
        prior: Prior,
        *,
        pi0: float = 0.5,
        seed: Optional[int] = None,
        record_pi_t: bool = False,
        **_: Any,
    ) -> Dict[str, Any]:
        """Run SMC under both outcomes and return posterior P(Y=1 | h)."""
        rng = np.random.default_rng(seed)
        seed0 = int(rng.integers(0, 2**31 - 1))
        seed1 = int(rng.integers(0, 2**31 - 1))

        res0 = self.marginal_likelihood(
            dx, v, 0, model, prior, seed=seed0, record_pi_t=record_pi_t,
        )
        res1 = self.marginal_likelihood(
            dx, v, 1, model, prior, seed=seed1, record_pi_t=record_pi_t,
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
            cum1 = np.cumsum(res1["log_inc"])
            cum0 = np.cumsum(res0["log_inc"])
            log_post_odds_t = logit_pi0 + cum1 - cum0
            out["pi_t"] = np.where(
                log_post_odds_t >= 0,
                1.0 / (1.0 + np.exp(-np.clip(log_post_odds_t, -500, 500))),
                np.exp(np.clip(log_post_odds_t, -500, 500)) /
                    (1.0 + np.exp(np.clip(log_post_odds_t, -500, 500))),
            )
            out["log_BF_t"] = cum1 - cum0
        return out
