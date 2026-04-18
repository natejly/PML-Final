"""Synthetic data generator for the Gaussian latent-type model (Example 3.1).

Produces histories (Delta x_{1:T}, v_{1:T}) under a known outcome y and known
nuisance parameters theta. Used by the synthetic sanity notebook to reproduce
the four experiments from Section 5.2.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional

import numpy as np


# Paper defaults (Section 5.2.1).
DEFAULT_THETA: Dict[str, np.ndarray] = dict(
    omega=np.array([0.4, 0.4, 0.2]),
    gamma=np.array([
        [0.0, 0.5],
        [0.0, 0.0],
        [0.0, 0.3],
    ]),
    mu1=np.array(0.5), lam1=np.array(0.1),
    sigma1=np.array(0.3), kappa1=np.array(0.05),
    sigma2=np.array(0.5),
    mu3=np.array(0.3), tau3=np.array(5.0),
    sigma3=np.array(0.4), nu=np.array(5.0),
)

DEFAULT_VOLUME_SHAPE = 2.0
DEFAULT_VOLUME_SCALE = 0.5


def sample_volumes(T: int, rng: np.random.Generator,
                   shape: float = DEFAULT_VOLUME_SHAPE,
                   scale: float = DEFAULT_VOLUME_SCALE) -> np.ndarray:
    """V_t ~ iid Gamma(shape, scale)."""
    return rng.gamma(shape, scale, size=T)


def _hard_indicator(v: np.ndarray, tau: float) -> np.ndarray:
    return (v > tau).astype(np.float64)


def simulate_increments(v: np.ndarray, y: int, theta: Dict[str, np.ndarray],
                        rng: np.random.Generator) -> np.ndarray:
    """Simulate Delta x_{1:T} from the volume-gated mixture under outcome y.

    For each t we draw a type k_t ~ Categorical(rho(v_t; omega, gamma)) and
    then draw Delta x_t from the type-k_t conditional density.
    """
    T = v.shape[0]
    K = 3

    log_omega = np.log(theta["omega"])
    g0 = theta["gamma"][:, 0]
    g1 = theta["gamma"][:, 1]
    log1pv = np.log1p(v)[:, None]
    logits = log_omega[None, :] + g0[None, :] + g1[None, :] * log1pv
    logits -= logits.max(axis=-1, keepdims=True)
    rho = np.exp(logits)
    rho /= rho.sum(axis=-1, keepdims=True)

    types = np.array([rng.choice(K, p=rho[t]) for t in range(T)])

    sign = 2 * y - 1

    m1 = float(theta["mu1"]) * sign * (1.0 - np.exp(-float(theta["lam1"]) * v))
    s1 = float(theta["sigma1"]) / np.sqrt(1.0 + float(theta["kappa1"]) * v)

    m2 = np.zeros(T)
    s2 = np.full(T, float(theta["sigma2"]))

    active = _hard_indicator(v, float(theta["tau3"]))
    m3 = -float(theta["mu3"]) * sign * active
    s3 = np.full(T, float(theta["sigma3"]))
    nu = float(theta["nu"])

    dx = np.empty(T)
    for t in range(T):
        k = types[t]
        if k == 0:
            dx[t] = rng.normal(m1[t], s1[t])
        elif k == 1:
            dx[t] = rng.normal(m2[t], s2[t])
        else:
            dx[t] = m3[t] + s3[t] * rng.standard_t(nu)
    return dx


def simulate_history(T: int, y_true: int = 1,
                     theta: Optional[Dict[str, np.ndarray]] = None,
                     volumes: Optional[np.ndarray] = None,
                     rng: Optional[np.random.Generator] = None,
                     seed: Optional[int] = None
                     ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Simulate one synthetic price-volume history.

    Returns (dx, v, y_true) where dx is the log-odds increment series of
    length T, v is the volume series of length T, and y_true is the outcome
    used.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if theta is None:
        theta = {k: np.array(v) for k, v in DEFAULT_THETA.items()}
    if volumes is None:
        v = sample_volumes(T, rng)
    else:
        v = np.asarray(volumes, dtype=np.float64)
        assert v.shape == (T,)
    dx = simulate_increments(v, y_true, theta, rng)
    return dx, v, y_true
