"""Prior + bijector for the burst-robust mispricing log-volume AR model.

`BurstMispricingLogARVolModel` uses the base increment-parameter block plus

    alpha_v             in R
    phi_v               in (-1, 1)
    beta_misprice       >= 0
    sigma_v             > 0
    eta_burst           in (0, 1)
    sigma_burst_extra   > 0
    nu_burst            > 2

The burst component is deliberately Y-independent, so generic news/attention
volume spikes do not have to be interpreted as evidence about the outcome.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping

import numpy as np
from scipy.special import gammaln as _np_lgamma

from ..core import Prior
from .base_prior import BasePrior
from ..models.base_model import PARAM_NAMES as BASE_PARAM_NAMES


ALPHA_V_PRIOR_SD = 5.0
PHI_V_UNCONSTRAINED_PRIOR_SD = 1.0
BETA_MISPRICE_PRIOR_SD = 1.0
SIGMA_V_PRIOR_SD = 1.0
ETA_BURST_ALPHA = 1.0
ETA_BURST_BETA = 9.0
SIGMA_BURST_EXTRA_PRIOR_SD = 3.0
NU_BURST_FLOOR = 2.0
NU_BURST_GAMMA_SHAPE = 2.0
NU_BURST_GAMMA_SCALE = 2.0


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


def _sigmoid(z):
    if _is_torch(z):
        import torch

        return torch.sigmoid(z)
    return 1.0 / (1.0 + np.exp(-z))


def _log_sigmoid_jac(z):
    if _is_torch(z):
        import torch

        return torch.nn.functional.logsigmoid(z) + torch.nn.functional.logsigmoid(-z)
    return -_softplus(-z) - _softplus(z)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-10, 1.0 - 1e-10)
    return np.log(p / (1.0 - p))


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


def _beta_logpdf_np(x, alpha: float, beta: float):
    x = np.asarray(x, dtype=np.float64)
    log_b = _np_lgamma(alpha) + _np_lgamma(beta) - _np_lgamma(alpha + beta)
    out = (alpha - 1.0) * np.log(x) + (beta - 1.0) * np.log1p(-x) - log_b
    return np.where((0.0 < x) & (x < 1.0), out, -np.inf)


def _gamma_shifted_logpdf_np(x, shape: float, scale: float, shift: float):
    z = np.asarray(x, dtype=np.float64) - shift
    out = (
        (shape - 1.0) * np.log(z)
        - z / scale
        - shape * math.log(scale)
        - _np_lgamma(shape)
    )
    return np.where(z > 0.0, out, -np.inf)


def log_prior(
    theta: Mapping[str, Any],
    base: BasePrior,
    alpha_v_sd: float = ALPHA_V_PRIOR_SD,
    phi_v_unconstrained_sd: float = PHI_V_UNCONSTRAINED_PRIOR_SD,
    beta_misprice_sd: float = BETA_MISPRICE_PRIOR_SD,
    sigma_v_sd: float = SIGMA_V_PRIOR_SD,
    eta_burst_alpha: float = ETA_BURST_ALPHA,
    eta_burst_beta: float = ETA_BURST_BETA,
    sigma_burst_extra_sd: float = SIGMA_BURST_EXTRA_PRIOR_SD,
    nu_burst_gamma_shape: float = NU_BURST_GAMMA_SHAPE,
    nu_burst_gamma_scale: float = NU_BURST_GAMMA_SCALE,
) -> float:
    """Constrained-space log prior for theta = (theta_base, theta_volume)."""
    alpha = float(np.asarray(theta["alpha_v"]))
    phi = float(np.asarray(theta["phi_v"]))
    beta = float(np.asarray(theta["beta_misprice"]))
    sigma = float(np.asarray(theta["sigma_v"]))
    eta = float(np.asarray(theta["eta_burst"]))
    sigma_extra = float(np.asarray(theta["sigma_burst_extra"]))
    nu = float(np.asarray(theta["nu_burst"]))

    if not (-1.0 < phi < 1.0):
        return float("-inf")

    z_phi = float(_atanh(np.asarray(phi)))
    lp = float(base.log_prior(theta))
    lp += float(_normal_logpdf_np(alpha, 0.0, alpha_v_sd))
    lp += float(_normal_logpdf_np(z_phi, 0.0, phi_v_unconstrained_sd))
    lp -= math.log(max(1.0 - phi * phi, 1e-12))
    lp += float(_halfnormal_logpdf_np(beta, beta_misprice_sd))
    lp += float(_halfnormal_logpdf_np(sigma, sigma_v_sd))
    lp += float(_beta_logpdf_np(eta, eta_burst_alpha, eta_burst_beta))
    lp += float(_halfnormal_logpdf_np(sigma_extra, sigma_burst_extra_sd))
    lp += float(_gamma_shifted_logpdf_np(
        nu,
        nu_burst_gamma_shape,
        nu_burst_gamma_scale,
        NU_BURST_FLOOR,
    ))
    return lp


def log_prior_batched(
    theta: Mapping[str, np.ndarray],
    base: BasePrior,
    alpha_v_sd: float = ALPHA_V_PRIOR_SD,
    phi_v_unconstrained_sd: float = PHI_V_UNCONSTRAINED_PRIOR_SD,
    beta_misprice_sd: float = BETA_MISPRICE_PRIOR_SD,
    sigma_v_sd: float = SIGMA_V_PRIOR_SD,
    eta_burst_alpha: float = ETA_BURST_ALPHA,
    eta_burst_beta: float = ETA_BURST_BETA,
    sigma_burst_extra_sd: float = SIGMA_BURST_EXTRA_PRIOR_SD,
    nu_burst_gamma_shape: float = NU_BURST_GAMMA_SHAPE,
    nu_burst_gamma_scale: float = NU_BURST_GAMMA_SCALE,
) -> np.ndarray:
    alpha = np.asarray(theta["alpha_v"], dtype=np.float64)
    phi = np.asarray(theta["phi_v"], dtype=np.float64)
    beta = np.asarray(theta["beta_misprice"], dtype=np.float64)
    sigma = np.asarray(theta["sigma_v"], dtype=np.float64)
    eta = np.asarray(theta["eta_burst"], dtype=np.float64)
    sigma_extra = np.asarray(theta["sigma_burst_extra"], dtype=np.float64)
    nu = np.asarray(theta["nu_burst"], dtype=np.float64)
    z_phi = _atanh(phi)

    out = base.log_prior_batched(theta)
    out = out + _normal_logpdf_np(alpha, 0.0, alpha_v_sd)
    out = out + _normal_logpdf_np(z_phi, 0.0, phi_v_unconstrained_sd)
    out = out - np.log(np.maximum(1.0 - phi * phi, 1e-12))
    out = out + _halfnormal_logpdf_np(beta, beta_misprice_sd)
    out = out + _halfnormal_logpdf_np(sigma, sigma_v_sd)
    out = out + _beta_logpdf_np(eta, eta_burst_alpha, eta_burst_beta)
    out = out + _halfnormal_logpdf_np(sigma_extra, sigma_burst_extra_sd)
    out = out + _gamma_shifted_logpdf_np(
        nu,
        nu_burst_gamma_shape,
        nu_burst_gamma_scale,
        NU_BURST_FLOOR,
    )
    out = np.where((-1.0 < phi) & (phi < 1.0), out, -np.inf)
    return out


class BurstMispricingLogARVolPrior(Prior):
    """Prior + bijector compatible with `BurstMispricingLogARVolModel`."""

    PARAM_NAMES = tuple(BASE_PARAM_NAMES) + (
        "alpha_v",
        "phi_v",
        "beta_misprice",
        "sigma_v",
        "eta_burst",
        "sigma_burst_extra",
        "nu_burst",
    )

    def __init__(
        self,
        base_prior: BasePrior | None = None,
        alpha_v_sd: float = ALPHA_V_PRIOR_SD,
        phi_v_unconstrained_sd: float = PHI_V_UNCONSTRAINED_PRIOR_SD,
        beta_misprice_sd: float = BETA_MISPRICE_PRIOR_SD,
        sigma_v_sd: float = SIGMA_V_PRIOR_SD,
        eta_burst_alpha: float = ETA_BURST_ALPHA,
        eta_burst_beta: float = ETA_BURST_BETA,
        sigma_burst_extra_sd: float = SIGMA_BURST_EXTRA_PRIOR_SD,
        nu_burst_gamma_shape: float = NU_BURST_GAMMA_SHAPE,
        nu_burst_gamma_scale: float = NU_BURST_GAMMA_SCALE,
    ):
        self.base = base_prior or BasePrior()
        self.alpha_v_sd = float(alpha_v_sd)
        self.phi_v_unconstrained_sd = float(phi_v_unconstrained_sd)
        self.beta_misprice_sd = float(beta_misprice_sd)
        self.sigma_v_sd = float(sigma_v_sd)
        self.eta_burst_alpha = float(eta_burst_alpha)
        self.eta_burst_beta = float(eta_burst_beta)
        self.sigma_burst_extra_sd = float(sigma_burst_extra_sd)
        self.nu_burst_gamma_shape = float(nu_burst_gamma_shape)
        self.nu_burst_gamma_scale = float(nu_burst_gamma_scale)
        self.UNCONSTRAINED_DIM = self.base.UNCONSTRAINED_DIM + 7

    def sample(self, rng: np.random.Generator, n: int = 1) -> Dict[str, np.ndarray]:
        n = int(n)
        base_theta = self.base.sample(rng, n)
        alpha = rng.normal(0.0, self.alpha_v_sd, size=n).astype(np.float64)
        z_phi = rng.normal(0.0, self.phi_v_unconstrained_sd, size=n).astype(np.float64)
        phi = np.tanh(z_phi)
        beta = np.abs(rng.normal(0.0, self.beta_misprice_sd, size=n)).astype(np.float64)
        sigma = np.abs(rng.normal(0.0, self.sigma_v_sd, size=n)).astype(np.float64)
        eta = rng.beta(self.eta_burst_alpha, self.eta_burst_beta, size=n).astype(np.float64)
        sigma_extra = np.abs(rng.normal(0.0, self.sigma_burst_extra_sd, size=n)).astype(np.float64)
        nu = NU_BURST_FLOOR + rng.gamma(
            self.nu_burst_gamma_shape,
            self.nu_burst_gamma_scale,
            size=n,
        ).astype(np.float64)
        return {
            **base_theta,
            "alpha_v": alpha,
            "phi_v": phi,
            "beta_misprice": beta,
            "sigma_v": sigma,
            "eta_burst": eta,
            "sigma_burst_extra": sigma_extra,
            "nu_burst": nu,
        }

    def transform(self, z):
        """Map unconstrained z (..., d+7) -> theta dict + log|det dtheta/dz|."""
        base_dim = self.base.UNCONSTRAINED_DIM
        base_z = z[..., :base_dim]
        vol_z = z[..., base_dim:]

        base_theta, base_log_jac = self.base.transform(base_z)
        alpha_z = vol_z[..., 0]
        phi_z = vol_z[..., 1]
        beta_z = vol_z[..., 2]
        sigma_z = vol_z[..., 3]
        eta_z = vol_z[..., 4]
        sigma_extra_z = vol_z[..., 5]
        nu_z = vol_z[..., 6]

        alpha_v = alpha_z
        phi_v = _tanh(phi_z)
        beta_misprice = _softplus(beta_z)
        sigma_v = _softplus(sigma_z)
        eta_burst = _sigmoid(eta_z)
        sigma_burst_extra = _softplus(sigma_extra_z)
        nu_burst = NU_BURST_FLOOR + _softplus(nu_z)

        log_jac = (
            base_log_jac
            + _log_tanh_jac(phi_z)
            + _log_softplus_jac(beta_z)
            + _log_softplus_jac(sigma_z)
            + _log_sigmoid_jac(eta_z)
            + _log_softplus_jac(sigma_extra_z)
            + _log_softplus_jac(nu_z)
        )
        return {
            **base_theta,
            "alpha_v": alpha_v,
            "phi_v": phi_v,
            "beta_misprice": beta_misprice,
            "sigma_v": sigma_v,
            "eta_burst": eta_burst,
            "sigma_burst_extra": sigma_burst_extra,
            "nu_burst": nu_burst,
        }, log_jac

    def to_unconstrained(self, theta: Mapping[str, Any]) -> np.ndarray:
        base_z = self.base.to_unconstrained(theta)
        alpha = np.asarray(theta["alpha_v"], dtype=np.float64)
        phi = np.asarray(theta["phi_v"], dtype=np.float64)
        beta = np.asarray(theta["beta_misprice"], dtype=np.float64)
        sigma = np.asarray(theta["sigma_v"], dtype=np.float64)
        eta = np.asarray(theta["eta_burst"], dtype=np.float64)
        sigma_extra = np.asarray(theta["sigma_burst_extra"], dtype=np.float64)
        nu = np.asarray(theta["nu_burst"], dtype=np.float64)

        alpha_z = alpha.reshape(alpha.shape + (1,))
        phi_z = _atanh(phi).reshape(phi.shape + (1,))
        beta_z = _softplus_inv(beta).reshape(beta.shape + (1,))
        sigma_z = _softplus_inv(sigma).reshape(sigma.shape + (1,))
        eta_z = _logit(eta).reshape(eta.shape + (1,))
        sigma_extra_z = _softplus_inv(sigma_extra).reshape(sigma_extra.shape + (1,))
        nu_z = _softplus_inv(nu - NU_BURST_FLOOR).reshape(nu.shape + (1,))
        vol_z = np.concatenate(
            [alpha_z, phi_z, beta_z, sigma_z, eta_z, sigma_extra_z, nu_z],
            axis=-1,
        )
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
        eta_z = vol_z[..., 4]
        sigma_extra_z = vol_z[..., 5]
        nu_z = vol_z[..., 6]

        alpha_const = 0.5 * math.log(2.0 * math.pi) + math.log(self.alpha_v_sd)
        phi_const = 0.5 * math.log(2.0 * math.pi) + math.log(self.phi_v_unconstrained_sd)
        alpha_lp = -0.5 * (alpha_z / self.alpha_v_sd) ** 2 - alpha_const
        phi_lp = -0.5 * (phi_z / self.phi_v_unconstrained_sd) ** 2 - phi_const

        beta = _softplus(beta_z)
        sigma = _softplus(sigma_z)
        eta = _sigmoid(eta_z)
        sigma_extra = _softplus(sigma_extra_z)
        nu_shift = _softplus(nu_z)

        halfnormal_const = lambda sd: 0.5 * math.log(2.0 / math.pi) - math.log(sd)
        beta_lp = halfnormal_const(self.beta_misprice_sd) - 0.5 * (beta / self.beta_misprice_sd) ** 2
        sigma_lp = halfnormal_const(self.sigma_v_sd) - 0.5 * (sigma / self.sigma_v_sd) ** 2
        sigma_extra_lp = (
            halfnormal_const(self.sigma_burst_extra_sd)
            - 0.5 * (sigma_extra / self.sigma_burst_extra_sd) ** 2
        )

        log_b = (
            _np_lgamma(self.eta_burst_alpha)
            + _np_lgamma(self.eta_burst_beta)
            - _np_lgamma(self.eta_burst_alpha + self.eta_burst_beta)
        )
        if _is_torch(eta):
            eta_lp = (
                (self.eta_burst_alpha - 1.0) * eta.log()
                + (self.eta_burst_beta - 1.0) * (1.0 - eta).log()
                - log_b
            )
            log_nu_shift = nu_shift.log()
        else:
            eta_lp = (
                (self.eta_burst_alpha - 1.0) * np.log(eta)
                + (self.eta_burst_beta - 1.0) * np.log1p(-eta)
                - log_b
            )
            log_nu_shift = np.log(nu_shift)

        nu_lp = (
            (self.nu_burst_gamma_shape - 1.0) * log_nu_shift
            - nu_shift / self.nu_burst_gamma_scale
            - self.nu_burst_gamma_shape * math.log(self.nu_burst_gamma_scale)
            - _np_lgamma(self.nu_burst_gamma_shape)
        )

        return (
            base_lp
            + alpha_lp
            + phi_lp
            + beta_lp
            + sigma_lp
            + eta_lp
            + sigma_extra_lp
            + nu_lp
            + _log_softplus_jac(beta_z)
            + _log_softplus_jac(sigma_z)
            + _log_sigmoid_jac(eta_z)
            + _log_softplus_jac(sigma_extra_z)
            + _log_softplus_jac(nu_z)
        )

    def log_prior(self, theta: Mapping[str, Any]) -> float:
        return log_prior(
            theta,
            self.base,
            self.alpha_v_sd,
            self.phi_v_unconstrained_sd,
            self.beta_misprice_sd,
            self.sigma_v_sd,
            self.eta_burst_alpha,
            self.eta_burst_beta,
            self.sigma_burst_extra_sd,
            self.nu_burst_gamma_shape,
            self.nu_burst_gamma_scale,
        )

    def log_prior_batched(self, theta: Mapping[str, np.ndarray]) -> np.ndarray:
        return log_prior_batched(
            theta,
            self.base,
            self.alpha_v_sd,
            self.phi_v_unconstrained_sd,
            self.beta_misprice_sd,
            self.sigma_v_sd,
            self.eta_burst_alpha,
            self.eta_burst_beta,
            self.sigma_burst_extra_sd,
            self.nu_burst_gamma_shape,
            self.nu_burst_gamma_scale,
        )
