"""Prior + bijector for the Gaussian Markov volume model.

`GaussianVolModel` uses the base increment-parameter block *and* one extra
volume parameter `sigma_v`:

    theta = (theta_base, sigma_v),  sigma_v > 0

This prior therefore composes:

  * `BasePrior` on `theta_base`
  * independent HalfNormal prior on `sigma_v`

The unconstrained parameterization appends one scalar to the base prior's
unconstrained vector and maps it with softplus.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping

import numpy as np

from ..core import Prior
from .base_prior import BasePrior
from ..models.base_model import PARAM_NAMES as BASE_PARAM_NAMES


SIGMA_V_PRIOR_SD = 1.0


def _is_torch(x: Any) -> bool:
    try:
        import torch

        return isinstance(x, torch.Tensor)
    except ImportError:
        return False


def _softplus(z):
    if _is_torch(z):
        import torch

        return torch.nn.functional.softplus(z)
    return np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))


def _softplus_inv(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.log(np.expm1(np.clip(x, 1e-10, None)))


def _log_softplus_jac(z):
    # log softplus'(z) = log sigmoid(z) = -softplus(-z)
    return -_softplus(-z)


def _halfnormal_logpdf_np(x: float, sd: float) -> float:
    if x < 0:
        return float("-inf")
    return 0.5 * math.log(2.0 / math.pi) - math.log(sd) - 0.5 * (x / sd) ** 2


def log_prior(theta: Mapping[str, Any], base: BasePrior) -> float:
    """Constrained-space log prior for theta = (theta_base, sigma_v)."""
    sigma_v = float(np.asarray(theta["sigma_v"]))
    return float(base.log_prior(theta) + _halfnormal_logpdf_np(sigma_v, SIGMA_V_PRIOR_SD))


def log_prior_batched(theta: Mapping[str, np.ndarray], base: BasePrior) -> np.ndarray:
    sigma = np.asarray(theta["sigma_v"], dtype=np.float64)
    const = 0.5 * math.log(2.0 / math.pi) - math.log(SIGMA_V_PRIOR_SD)
    out = const - 0.5 * (sigma / SIGMA_V_PRIOR_SD) ** 2
    out = np.where(sigma >= 0, out, -np.inf)
    return base.log_prior_batched(theta) + out


class GaussianVolPrior(Prior):
    """Prior + bijector compatible with `GaussianVolModel`.

    The prior factorizes as:
        pi(theta_base, sigma_v) = pi_base(theta_base) * pi_sigma(sigma_v),
        sigma_v ~ HalfNormal(SIGMA_V_PRIOR_SD).
    """

    PARAM_NAMES = tuple(BASE_PARAM_NAMES) + ("sigma_v",)

    def __init__(self, base_prior: BasePrior | None = None, sigma_v_sd: float = SIGMA_V_PRIOR_SD):
        self.base = base_prior or BasePrior()
        self.sigma_v_sd = float(sigma_v_sd)
        self.UNCONSTRAINED_DIM = self.base.UNCONSTRAINED_DIM + 1

    def sample(self, rng: np.random.Generator, n: int = 1) -> Dict[str, np.ndarray]:
        n = int(n)
        base_theta = self.base.sample(rng, n)
        sigma = np.abs(rng.normal(0.0, self.sigma_v_sd, size=n)).astype(np.float64)
        return {**base_theta, "sigma_v": sigma}

    def transform(self, z):
        """Map unconstrained z (..., d+1) -> theta dict + log|det dtheta/dz|."""
        base_dim = self.base.UNCONSTRAINED_DIM
        base_z = z[..., :base_dim]
        sigma_z = z[..., base_dim]

        base_theta, base_log_jac = self.base.transform(base_z)
        sigma_v = _softplus(sigma_z)
        sigma_log_jac = _log_softplus_jac(sigma_z)

        return {**base_theta, "sigma_v": sigma_v}, (base_log_jac + sigma_log_jac)

    def to_unconstrained(self, theta: Mapping[str, Any]) -> np.ndarray:
        base_z = self.base.to_unconstrained(theta)
        sigma = np.asarray(theta["sigma_v"], dtype=np.float64)
        sigma_z = _softplus_inv(sigma).reshape(sigma.shape + (1,))
        return np.concatenate([base_z, sigma_z], axis=-1)

    def log_prior_unconstrained(self, z) -> Any:
        """log pi(z) = log Pi(theta(z)) + log|det dtheta/dz|."""
        base_dim = self.base.UNCONSTRAINED_DIM
        base_z = z[..., :base_dim]
        sigma_z = z[..., base_dim]

        base_lp = self.base.log_prior_unconstrained(base_z)

        sigma_v = _softplus(sigma_z)
        const = 0.5 * math.log(2.0 / math.pi) - math.log(self.sigma_v_sd)
        sigma_lp = const - 0.5 * (sigma_v / self.sigma_v_sd) ** 2

        if _is_torch(sigma_v):
            import torch

            sigma_lp = torch.where(sigma_v >= 0, sigma_lp, torch.full_like(sigma_lp, float("-inf")))
        else:
            sigma_lp = np.where(sigma_v >= 0, sigma_lp, -np.inf)

        return base_lp + sigma_lp + _log_softplus_jac(sigma_z)

    def log_prior(self, theta: Mapping[str, Any]) -> float:
        return log_prior(theta, self.base)

    def log_prior_batched(self, theta: Mapping[str, np.ndarray]) -> np.ndarray:
        return log_prior_batched(theta, self.base)
