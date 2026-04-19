"""Priors and unconstrained transform for a simple Markovian volume model.

This module is intentionally minimal. It parameterizes volume dynamics with one
parameter:

    sigma_v > 0

and uses the transition model (implemented in `model_vol.py`):

    v_t | v_{t-1}, sigma_v ~ Normal(v_{t-1}, sigma_v^2),  t >= 2.

Prior:

    sigma_v ~ HalfNormal(1.0)

A softplus bijection is used so SMC/VI can work in unconstrained space.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
from scipy.special import gammaln as _np_lgamma  # kept for consistency/style


PARAM_NAMES_VOL = ("sigma_v",)
UNCONSTRAINED_DIM_VOL = 1
SIGMA_V_PRIOR_SD = 1.0


def _is_torch(x: Any) -> bool:
    try:
        import torch

        return isinstance(x, torch.Tensor)
    except ImportError:
        return False


def _backend(x: Any):
    if _is_torch(x):
        import torch

        return torch
    return np


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


def log_prior(theta: Dict[str, Any]) -> float:
    """Log prior density on constrained parameters."""
    sigma_v = float(np.asarray(theta["sigma_v"]))
    return float(_halfnormal_logpdf_np(sigma_v, SIGMA_V_PRIOR_SD))


def log_prior_batched(theta: Dict[str, np.ndarray]) -> np.ndarray:
    sigma = np.asarray(theta["sigma_v"], dtype=np.float64)
    const = 0.5 * math.log(2.0 / math.pi) - math.log(SIGMA_V_PRIOR_SD)
    out = const - 0.5 * (sigma / SIGMA_V_PRIOR_SD) ** 2
    out = np.where(sigma >= 0, out, -np.inf)
    return out


def sample_prior(rng: np.random.Generator) -> Dict[str, np.ndarray]:
    return {"sigma_v": np.array(abs(rng.normal(0.0, SIGMA_V_PRIOR_SD)), dtype=np.float64)}


def sample_prior_batched(rng: np.random.Generator, n: int) -> Dict[str, np.ndarray]:
    sigma = abs(rng.normal(0.0, SIGMA_V_PRIOR_SD, size=n)).astype(np.float64)
    return {"sigma_v": sigma}


def transform(z):
    """Map unconstrained z (..., 1) -> theta dict + log|det dtheta/dz|."""
    sigma_v = _softplus(z[..., 0])
    log_jac = _log_softplus_jac(z[..., 0])
    return {"sigma_v": sigma_v}, log_jac


def to_unconstrained(theta: Dict[str, Any]) -> np.ndarray:
    sigma = np.asarray(theta["sigma_v"], dtype=np.float64)
    return _softplus_inv(sigma).reshape(sigma.shape + (1,))


def log_prior_unconstrained(z) -> Any:
    """log pi(z) = log Pi(theta(z)) + log|det dtheta/dz|."""
    theta, log_jac = transform(z)

    sigma = theta["sigma_v"]
    const = 0.5 * math.log(2.0 / math.pi) - math.log(SIGMA_V_PRIOR_SD)
    lp = const - 0.5 * (sigma / SIGMA_V_PRIOR_SD) ** 2

    if _is_torch(sigma):
        import torch

        lp = torch.where(sigma >= 0, lp, torch.full_like(lp, float("-inf")))
    else:
        lp = np.where(sigma >= 0, lp, -np.inf)

    return lp + log_jac
