"""Shared machinery for joint-Markov synthetic samplers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from ..synthetic import DEFAULT_THETA


DEFAULT_ALPHA_V = float(np.log1p(1.0))
DEFAULT_PHI_V = 0.70
DEFAULT_SIGMA_V = 0.25
DEFAULT_BETA_MISPRICE = 0.45
DEFAULT_BETA_REVERSAL = 0.45
DEFAULT_BETA_MOMENTUM = 0.20
DEFAULT_ETA_BURST = 0.08
DEFAULT_SIGMA_BURST_EXTRA = 0.70
DEFAULT_NU_BURST = 5.0
U_MIN = 0.0


def copy_theta(theta: Mapping[str, object]) -> dict[str, np.ndarray]:
    return {k: np.array(v, dtype=np.float64, copy=True) for k, v in theta.items()}


def base_theta() -> dict[str, np.ndarray]:
    return copy_theta(DEFAULT_THETA)


def scalar(theta: Mapping[str, object], name: str) -> float:
    arr = np.asarray(theta[name], dtype=np.float64)
    if arr.size != 1:
        raise ValueError(f"samplers expect scalar theta[{name!r}], got shape {arr.shape}")
    return float(arr.reshape(()))


def positive_part(x: float) -> float:
    return max(float(x), 0.0)


def reversal_pressure(dx_prev: float, y: int) -> float:
    sign = 2.0 * int(y) - 1.0
    return positive_part(-sign * dx_prev)


def momentum_pressure(dx_prev: float, y: int) -> float:
    sign = 2.0 * int(y) - 1.0
    return positive_part(sign * dx_prev)


def draw_nonnegative_normal_u(mean: float, scale: float, rng: np.random.Generator,
                              max_attempts: int) -> float:
    for _ in range(max_attempts):
        u = float(rng.normal(mean, scale))
        if u >= U_MIN:
            return u
    return U_MIN


def draw_nonnegative_student_u(mean: float, scale: float, nu: float,
                               rng: np.random.Generator,
                               max_attempts: int) -> float:
    for _ in range(max_attempts):
        u = float(mean + scale * rng.standard_t(nu))
        if u >= U_MIN:
            return u
    return U_MIN


def _gate_probs(v: float, theta: Mapping[str, object]) -> np.ndarray:
    omega = np.asarray(theta["omega"], dtype=np.float64)
    gamma = np.asarray(theta["gamma"], dtype=np.float64)
    if omega.shape != (3,):
        raise ValueError(f"samplers expect theta['omega'] shape (3,), got {omega.shape}")
    if gamma.shape != (3, 2):
        raise ValueError(f"samplers expect theta['gamma'] shape (3, 2), got {gamma.shape}")
    logits = np.log(omega) + gamma[:, 0] + gamma[:, 1] * np.log1p(v)
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    return probs / np.sum(probs)


def sample_increment(v: float, y: int, theta: Mapping[str, object],
                     rng: np.random.Generator) -> float:
    probs = _gate_probs(v, theta)
    k = int(rng.choice(3, p=probs))
    sign = 2.0 * int(y) - 1.0

    if k == 0:
        mu1 = scalar(theta, "mu1")
        lam1 = scalar(theta, "lam1")
        sigma1 = scalar(theta, "sigma1")
        kappa1 = scalar(theta, "kappa1")
        mean = mu1 * sign * (1.0 - np.exp(-lam1 * v))
        scale = sigma1 / np.sqrt(1.0 + kappa1 * v)
        return float(rng.normal(mean, scale))

    if k == 1:
        sigma2 = scalar(theta, "sigma2")
        return float(rng.normal(0.0, sigma2))

    mu3 = scalar(theta, "mu3")
    tau3 = scalar(theta, "tau3")
    sigma3 = scalar(theta, "sigma3")
    nu = scalar(theta, "nu")
    mean = -mu3 * sign * float(v > tau3)
    return float(mean + sigma3 * rng.standard_t(nu))


@dataclass
class JointMarkovSamplerBase:
    """Base class for synthetic joint-Markov forward samplers."""

    max_initial_u: Optional[float] = None
    max_resample_attempts: int = 100

    def default_theta(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def _next_u(self, u_prev: float, dx_prev: float, y: int,
                theta: Mapping[str, object], rng: np.random.Generator) -> float:
        raise NotImplementedError

    def initial_u(self, theta: Mapping[str, object]) -> float:
        u0 = max(scalar(theta, "alpha_v"), U_MIN)
        if self.max_initial_u is not None:
            u0 = min(u0, float(self.max_initial_u))
        return u0

    def sample(
        self,
        T: int,
        y: int = 1,
        theta: Optional[Mapping[str, object]] = None,
        *,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        initial_u: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Sample one synthetic history `(dx, v, y)`."""
        T = int(T)
        if T <= 0:
            raise ValueError("T must be positive")
        if rng is None:
            rng = np.random.default_rng(seed)
        if theta is None:
            theta = self.default_theta()

        dx = np.empty(T, dtype=np.float64)
        v = np.empty(T, dtype=np.float64)
        u = np.empty(T, dtype=np.float64)

        u[0] = max(float(initial_u), U_MIN) if initial_u is not None else self.initial_u(theta)
        v[0] = np.expm1(u[0])
        dx[0] = sample_increment(v[0], y, theta, rng)

        for t in range(1, T):
            u[t] = self._next_u(u[t - 1], dx[t - 1], y, theta, rng)
            v[t] = np.expm1(u[t])
            dx[t] = sample_increment(v[t], y, theta, rng)

        return dx, v, int(y)

    def sample_many(
        self,
        n: int,
        T: int,
        y: int = 1,
        theta: Optional[Mapping[str, object]] = None,
        *,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample `n` independent histories.

        Returns `(dx, v, y_arr)` with shapes `(n, T)`, `(n, T)`, `(n,)`.
        """
        n = int(n)
        if n <= 0:
            raise ValueError("n must be positive")
        rng = np.random.default_rng(seed)
        dxs = np.empty((n, int(T)), dtype=np.float64)
        vs = np.empty((n, int(T)), dtype=np.float64)
        ys = np.full(n, int(y), dtype=np.int64)
        for i in range(n):
            child_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
            dx, v, _ = self.sample(T, y, theta, rng=child_rng)
            dxs[i] = dx
            vs[i] = v
        return dxs, vs, ys
