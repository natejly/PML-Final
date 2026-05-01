"""Prior + bijector for the mispricing-reactive log-volume AR model.

`MispricingLogARVolModel` uses the base increment-parameter block plus

    alpha_v          in R
    phi_v            in (-1, 1)
    beta_misprice    >= 0
    sigma_v          > 0

The unconstrained parameterization appends four scalars to the base prior:

    alpha_v       = z_alpha
    phi_v         = tanh(z_phi)
    beta_misprice = softplus(z_beta)
    sigma_v       = softplus(z_sigma)

We place a Normal prior on alpha_v, a Normal prior on z_phi, and HalfNormal
priors on beta_misprice and sigma_v.  The nonnegative beta_misprice prior is
the outcome-orientation constraint for the volume dynamics.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping

import numpy as np

from ..core import Prior
from .base_prior import BasePrior
from ..models.base_model import PARAM_NAMES as BASE_PARAM_NAMES


ALPHA_V_PRIOR_SD = 5.0
PHI_V_UNCONSTRAINED_PRIOR_SD = 1.0
BETA_MISPRICE_PRIOR_SD = 1.0
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
    return -_softplus(-z)


def _tanh(z):
    if _is_torch(z):
        import torch

        return torch.tanh(z)
    return np.tanh(z)


def _atanh(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -1.0 + 1e-10, 1.0 - 1e-10)
    return 0.5 * np.log((1.0 + x) / (1.0 - x))


def _log_tanh_jac(z):
    phi = _tanh(z)
    if _is_torch(phi):
        return (1.0 - phi * phi).clamp(min=1e-12).log()
    return np.log(np.maximum(1.0 - phi * phi, 1e-12))


def _normal_logpdf_np(x, mean: float, sd: float):
    return -0.5 * ((x - mean) / sd) ** 2 - 0.5 * math.log(2.0 * math.pi) - math.log(sd)


def _halfnormal_logpdf_np(x, sd: float):
    x = np.asarray(x, dtype=np.float64)
    out = 0.5 * math.log(2.0 / math.pi) - math.log(sd) - 0.5 * (x / sd) ** 2
    return np.where(x >= 0, out, -np.inf)


def log_prior(
    theta: Mapping[str, Any],
    base: BasePrior,
    alpha_v_sd: float = ALPHA_V_PRIOR_SD,
    phi_v_unconstrained_sd: float = PHI_V_UNCONSTRAINED_PRIOR_SD,
    beta_misprice_sd: float = BETA_MISPRICE_PRIOR_SD,
    sigma_v_sd: float = SIGMA_V_PRIOR_SD,
) -> float:
    """Constrained-space log prior for theta = (theta_base, theta_volume)."""
    alpha = float(np.asarray(theta["alpha_v"]))
    phi = float(np.asarray(theta["phi_v"]))
    beta = float(np.asarray(theta["beta_misprice"]))
    sigma = float(np.asarray(theta["sigma_v"]))

    if not (-1.0 < phi < 1.0):
        return float("-inf")

    z_phi = float(_atanh(np.asarray(phi)))
    lp = float(base.log_prior(theta))
    lp += float(_normal_logpdf_np(alpha, 0.0, alpha_v_sd))
    # Induced density from z_phi ~ Normal(0, sd^2), phi = tanh(z_phi).
    lp += float(_normal_logpdf_np(z_phi, 0.0, phi_v_unconstrained_sd))
    lp -= math.log(max(1.0 - phi * phi, 1e-12))
    lp += float(_halfnormal_logpdf_np(beta, beta_misprice_sd))
    lp += float(_halfnormal_logpdf_np(sigma, sigma_v_sd))
    return lp


def log_prior_batched(
    theta: Mapping[str, np.ndarray],
    base: BasePrior,
    alpha_v_sd: float = ALPHA_V_PRIOR_SD,
    phi_v_unconstrained_sd: float = PHI_V_UNCONSTRAINED_PRIOR_SD,
    beta_misprice_sd: float = BETA_MISPRICE_PRIOR_SD,
    sigma_v_sd: float = SIGMA_V_PRIOR_SD,
) -> np.ndarray:
    alpha = np.asarray(theta["alpha_v"], dtype=np.float64)
    phi = np.asarray(theta["phi_v"], dtype=np.float64)
    beta = np.asarray(theta["beta_misprice"], dtype=np.float64)
    sigma = np.asarray(theta["sigma_v"], dtype=np.float64)
    z_phi = _atanh(phi)

    out = base.log_prior_batched(theta)
    out = out + _normal_logpdf_np(alpha, 0.0, alpha_v_sd)
    out = out + _normal_logpdf_np(z_phi, 0.0, phi_v_unconstrained_sd)
    out = out - np.log(np.maximum(1.0 - phi * phi, 1e-12))
    out = out + _halfnormal_logpdf_np(beta, beta_misprice_sd)
    out = out + _halfnormal_logpdf_np(sigma, sigma_v_sd)
    out = np.where((-1.0 < phi) & (phi < 1.0), out, -np.inf)
    return out


class MispricingLogARVolPrior(Prior):
    """Prior + bijector compatible with `MispricingLogARVolModel`."""

    PARAM_NAMES = tuple(BASE_PARAM_NAMES) + (
        "alpha_v",
        "phi_v",
        "beta_misprice",
        "sigma_v",
    )

    def __init__(
        self,
        base_prior: BasePrior | None = None,
        alpha_v_sd: float = ALPHA_V_PRIOR_SD,
        phi_v_unconstrained_sd: float = PHI_V_UNCONSTRAINED_PRIOR_SD,
        beta_misprice_sd: float = BETA_MISPRICE_PRIOR_SD,
        sigma_v_sd: float = SIGMA_V_PRIOR_SD,
    ):
        self.base = base_prior or BasePrior()
        self.alpha_v_sd = float(alpha_v_sd)
        self.phi_v_unconstrained_sd = float(phi_v_unconstrained_sd)
        self.beta_misprice_sd = float(beta_misprice_sd)
        self.sigma_v_sd = float(sigma_v_sd)
        self.UNCONSTRAINED_DIM = self.base.UNCONSTRAINED_DIM + 4

    def sample(self, rng: np.random.Generator, n: int = 1) -> Dict[str, np.ndarray]:
        n = int(n)
        base_theta = self.base.sample(rng, n)
        alpha = rng.normal(0.0, self.alpha_v_sd, size=n).astype(np.float64)
        z_phi = rng.normal(0.0, self.phi_v_unconstrained_sd, size=n).astype(np.float64)
        phi = np.tanh(z_phi)
        beta = np.abs(rng.normal(0.0, self.beta_misprice_sd, size=n)).astype(np.float64)
        sigma = np.abs(rng.normal(0.0, self.sigma_v_sd, size=n)).astype(np.float64)
        return {
            **base_theta,
            "alpha_v": alpha,
            "phi_v": phi,
            "beta_misprice": beta,
            "sigma_v": sigma,
        }

    def transform(self, z):
        """Map unconstrained z (..., d+4) -> theta dict + log|det dtheta/dz|."""
        base_dim = self.base.UNCONSTRAINED_DIM
        base_z = z[..., :base_dim]
        vol_z = z[..., base_dim:]

        base_theta, base_log_jac = self.base.transform(base_z)
        alpha_z = vol_z[..., 0]
        phi_z = vol_z[..., 1]
        beta_z = vol_z[..., 2]
        sigma_z = vol_z[..., 3]

        alpha_v = alpha_z
        phi_v = _tanh(phi_z)
        beta_misprice = _softplus(beta_z)
        sigma_v = _softplus(sigma_z)

        log_jac = (
            base_log_jac
            + _log_tanh_jac(phi_z)
            + _log_softplus_jac(beta_z)
            + _log_softplus_jac(sigma_z)
        )
        return {
            **base_theta,
            "alpha_v": alpha_v,
            "phi_v": phi_v,
            "beta_misprice": beta_misprice,
            "sigma_v": sigma_v,
        }, log_jac

    def to_unconstrained(self, theta: Mapping[str, Any]) -> np.ndarray:
        base_z = self.base.to_unconstrained(theta)
        alpha = np.asarray(theta["alpha_v"], dtype=np.float64)
        phi = np.asarray(theta["phi_v"], dtype=np.float64)
        beta = np.asarray(theta["beta_misprice"], dtype=np.float64)
        sigma = np.asarray(theta["sigma_v"], dtype=np.float64)

        alpha_z = alpha.reshape(alpha.shape + (1,))
        phi_z = _atanh(phi).reshape(phi.shape + (1,))
        beta_z = _softplus_inv(beta).reshape(beta.shape + (1,))
        sigma_z = _softplus_inv(sigma).reshape(sigma.shape + (1,))
        vol_z = np.concatenate([alpha_z, phi_z, beta_z, sigma_z], axis=-1)
        return np.concatenate([base_z, vol_z], axis=-1)

    def log_prior_unconstrained(self, z) -> Any:
        """log pi(z) = log Pi(theta(z)) + log|det dtheta/dz|."""
        base_dim = self.base.UNCONSTRAINED_DIM
        base_z = z[..., :base_dim]
        vol_z = z[..., base_dim:]

        base_lp = self.base.log_prior_unconstrained(base_z)
        alpha_z = vol_z[..., 0]
        phi_z = vol_z[..., 1]
        beta_z = vol_z[..., 2]
        sigma_z = vol_z[..., 3]

        alpha_const = 0.5 * math.log(2.0 * math.pi) + math.log(self.alpha_v_sd)
        phi_const = 0.5 * math.log(2.0 * math.pi) + math.log(self.phi_v_unconstrained_sd)
        alpha_lp = -0.5 * (alpha_z / self.alpha_v_sd) ** 2 - alpha_const
        phi_lp = -0.5 * (phi_z / self.phi_v_unconstrained_sd) ** 2 - phi_const

        beta = _softplus(beta_z)
        sigma = _softplus(sigma_z)
        beta_const = 0.5 * math.log(2.0 / math.pi) - math.log(self.beta_misprice_sd)
        sigma_const = 0.5 * math.log(2.0 / math.pi) - math.log(self.sigma_v_sd)
        beta_lp = beta_const - 0.5 * (beta / self.beta_misprice_sd) ** 2
        sigma_lp = sigma_const - 0.5 * (sigma / self.sigma_v_sd) ** 2

        if _is_torch(sigma):
            import torch

            neg_inf_beta = torch.full_like(beta_lp, float("-inf"))
            neg_inf_sigma = torch.full_like(sigma_lp, float("-inf"))
            beta_lp = torch.where(beta >= 0, beta_lp, neg_inf_beta)
            sigma_lp = torch.where(sigma >= 0, sigma_lp, neg_inf_sigma)
        else:
            beta_lp = np.where(beta >= 0, beta_lp, -np.inf)
            sigma_lp = np.where(sigma >= 0, sigma_lp, -np.inf)

        return (
            base_lp
            + alpha_lp
            + phi_lp
            + beta_lp
            + sigma_lp
            + _log_softplus_jac(beta_z)
            + _log_softplus_jac(sigma_z)
        )

    def log_prior(self, theta: Mapping[str, Any]) -> float:
        return log_prior(
            theta,
            self.base,
            self.alpha_v_sd,
            self.phi_v_unconstrained_sd,
            self.beta_misprice_sd,
            self.sigma_v_sd,
        )

    def log_prior_batched(self, theta: Mapping[str, np.ndarray]) -> np.ndarray:
        return log_prior_batched(
            theta,
            self.base,
            self.alpha_v_sd,
            self.phi_v_unconstrained_sd,
            self.beta_misprice_sd,
            self.sigma_v_sd,
        )
