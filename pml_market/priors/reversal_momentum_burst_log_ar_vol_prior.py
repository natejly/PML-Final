"""Prior + bijector for the reversal/momentum burst log-volume AR model.

`ReversalMomentumBurstLogARVolModel` uses the base increment-parameter block
plus

    alpha_v             in R
    phi_v               in (-1, 1)
    beta_reversal       >= 0
    beta_momentum       >= 0
    sigma_v             > 0
    eta_burst           in (0, 1)
    sigma_burst_extra   > 0
    nu_burst            > 2

The two nonnegative beta terms let the ordinary volume channel learn whether
volume follows contrarian correction, trend-following attention, or both.  The
burst channel remains Y-independent.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping

import numpy as np
from scipy.special import gammaln as _np_lgamma

from ..core import Prior
from .base_prior import BasePrior
from ..models.base_model import PARAM_NAMES as BASE_PARAM_NAMES
from .burst_mispricing_log_ar_vol_prior import (
    ALPHA_V_PRIOR_SD,
    BETA_MISPRICE_PRIOR_SD,
    ETA_BURST_ALPHA,
    ETA_BURST_BETA,
    NU_BURST_FLOOR,
    NU_BURST_GAMMA_SCALE,
    NU_BURST_GAMMA_SHAPE,
    PHI_V_UNCONSTRAINED_PRIOR_SD,
    SIGMA_BURST_EXTRA_PRIOR_SD,
    SIGMA_V_PRIOR_SD,
    _atanh,
    _beta_logpdf_np,
    _gamma_shifted_logpdf_np,
    _halfnormal_logpdf_np,
    _is_torch,
    _log_sigmoid_jac,
    _log_softplus_jac,
    _log_tanh_jac,
    _logit,
    _normal_logpdf_np,
    _sigmoid,
    _softplus,
    _softplus_inv,
    _tanh,
)


BETA_REVERSAL_PRIOR_SD = BETA_MISPRICE_PRIOR_SD
BETA_MOMENTUM_PRIOR_SD = BETA_MISPRICE_PRIOR_SD


def log_prior(
    theta: Mapping[str, Any],
    base: BasePrior,
    alpha_v_sd: float = ALPHA_V_PRIOR_SD,
    phi_v_unconstrained_sd: float = PHI_V_UNCONSTRAINED_PRIOR_SD,
    beta_reversal_sd: float = BETA_REVERSAL_PRIOR_SD,
    beta_momentum_sd: float = BETA_MOMENTUM_PRIOR_SD,
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
    beta_reversal = float(np.asarray(theta["beta_reversal"]))
    beta_momentum = float(np.asarray(theta["beta_momentum"]))
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
    lp += float(_halfnormal_logpdf_np(beta_reversal, beta_reversal_sd))
    lp += float(_halfnormal_logpdf_np(beta_momentum, beta_momentum_sd))
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
    beta_reversal_sd: float = BETA_REVERSAL_PRIOR_SD,
    beta_momentum_sd: float = BETA_MOMENTUM_PRIOR_SD,
    sigma_v_sd: float = SIGMA_V_PRIOR_SD,
    eta_burst_alpha: float = ETA_BURST_ALPHA,
    eta_burst_beta: float = ETA_BURST_BETA,
    sigma_burst_extra_sd: float = SIGMA_BURST_EXTRA_PRIOR_SD,
    nu_burst_gamma_shape: float = NU_BURST_GAMMA_SHAPE,
    nu_burst_gamma_scale: float = NU_BURST_GAMMA_SCALE,
) -> np.ndarray:
    alpha = np.asarray(theta["alpha_v"], dtype=np.float64)
    phi = np.asarray(theta["phi_v"], dtype=np.float64)
    beta_reversal = np.asarray(theta["beta_reversal"], dtype=np.float64)
    beta_momentum = np.asarray(theta["beta_momentum"], dtype=np.float64)
    sigma = np.asarray(theta["sigma_v"], dtype=np.float64)
    eta = np.asarray(theta["eta_burst"], dtype=np.float64)
    sigma_extra = np.asarray(theta["sigma_burst_extra"], dtype=np.float64)
    nu = np.asarray(theta["nu_burst"], dtype=np.float64)
    z_phi = _atanh(phi)

    out = base.log_prior_batched(theta)
    out = out + _normal_logpdf_np(alpha, 0.0, alpha_v_sd)
    out = out + _normal_logpdf_np(z_phi, 0.0, phi_v_unconstrained_sd)
    out = out - np.log(np.maximum(1.0 - phi * phi, 1e-12))
    out = out + _halfnormal_logpdf_np(beta_reversal, beta_reversal_sd)
    out = out + _halfnormal_logpdf_np(beta_momentum, beta_momentum_sd)
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


class ReversalMomentumBurstLogARVolPrior(Prior):
    """Prior + bijector compatible with `ReversalMomentumBurstLogARVolModel`."""

    PARAM_NAMES = tuple(BASE_PARAM_NAMES) + (
        "alpha_v",
        "phi_v",
        "beta_reversal",
        "beta_momentum",
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
        beta_reversal_sd: float = BETA_REVERSAL_PRIOR_SD,
        beta_momentum_sd: float = BETA_MOMENTUM_PRIOR_SD,
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
        self.beta_reversal_sd = float(beta_reversal_sd)
        self.beta_momentum_sd = float(beta_momentum_sd)
        self.sigma_v_sd = float(sigma_v_sd)
        self.eta_burst_alpha = float(eta_burst_alpha)
        self.eta_burst_beta = float(eta_burst_beta)
        self.sigma_burst_extra_sd = float(sigma_burst_extra_sd)
        self.nu_burst_gamma_shape = float(nu_burst_gamma_shape)
        self.nu_burst_gamma_scale = float(nu_burst_gamma_scale)
        self.UNCONSTRAINED_DIM = self.base.UNCONSTRAINED_DIM + 8

    def sample(self, rng: np.random.Generator, n: int = 1) -> Dict[str, np.ndarray]:
        n = int(n)
        base_theta = self.base.sample(rng, n)
        alpha = rng.normal(0.0, self.alpha_v_sd, size=n).astype(np.float64)
        z_phi = rng.normal(0.0, self.phi_v_unconstrained_sd, size=n).astype(np.float64)
        phi = np.tanh(z_phi)
        beta_reversal = np.abs(rng.normal(0.0, self.beta_reversal_sd, size=n)).astype(np.float64)
        beta_momentum = np.abs(rng.normal(0.0, self.beta_momentum_sd, size=n)).astype(np.float64)
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
            "beta_reversal": beta_reversal,
            "beta_momentum": beta_momentum,
            "sigma_v": sigma,
            "eta_burst": eta,
            "sigma_burst_extra": sigma_extra,
            "nu_burst": nu,
        }

    def transform(self, z):
        """Map unconstrained z (..., d+8) -> theta dict + log|det dtheta/dz|."""
        base_dim = self.base.UNCONSTRAINED_DIM
        base_z = z[..., :base_dim]
        vol_z = z[..., base_dim:]

        base_theta, base_log_jac = self.base.transform(base_z)
        alpha_z = vol_z[..., 0]
        phi_z = vol_z[..., 1]
        beta_reversal_z = vol_z[..., 2]
        beta_momentum_z = vol_z[..., 3]
        sigma_z = vol_z[..., 4]
        eta_z = vol_z[..., 5]
        sigma_extra_z = vol_z[..., 6]
        nu_z = vol_z[..., 7]

        alpha_v = alpha_z
        phi_v = _tanh(phi_z)
        beta_reversal = _softplus(beta_reversal_z)
        beta_momentum = _softplus(beta_momentum_z)
        sigma_v = _softplus(sigma_z)
        eta_burst = _sigmoid(eta_z)
        sigma_burst_extra = _softplus(sigma_extra_z)
        nu_burst = NU_BURST_FLOOR + _softplus(nu_z)

        log_jac = (
            base_log_jac
            + _log_tanh_jac(phi_z)
            + _log_softplus_jac(beta_reversal_z)
            + _log_softplus_jac(beta_momentum_z)
            + _log_softplus_jac(sigma_z)
            + _log_sigmoid_jac(eta_z)
            + _log_softplus_jac(sigma_extra_z)
            + _log_softplus_jac(nu_z)
        )
        return {
            **base_theta,
            "alpha_v": alpha_v,
            "phi_v": phi_v,
            "beta_reversal": beta_reversal,
            "beta_momentum": beta_momentum,
            "sigma_v": sigma_v,
            "eta_burst": eta_burst,
            "sigma_burst_extra": sigma_burst_extra,
            "nu_burst": nu_burst,
        }, log_jac

    def to_unconstrained(self, theta: Mapping[str, Any]) -> np.ndarray:
        base_z = self.base.to_unconstrained(theta)
        alpha = np.asarray(theta["alpha_v"], dtype=np.float64)
        phi = np.asarray(theta["phi_v"], dtype=np.float64)
        beta_reversal = np.asarray(theta["beta_reversal"], dtype=np.float64)
        beta_momentum = np.asarray(theta["beta_momentum"], dtype=np.float64)
        sigma = np.asarray(theta["sigma_v"], dtype=np.float64)
        eta = np.asarray(theta["eta_burst"], dtype=np.float64)
        sigma_extra = np.asarray(theta["sigma_burst_extra"], dtype=np.float64)
        nu = np.asarray(theta["nu_burst"], dtype=np.float64)

        alpha_z = alpha.reshape(alpha.shape + (1,))
        phi_z = _atanh(phi).reshape(phi.shape + (1,))
        beta_reversal_z = _softplus_inv(beta_reversal).reshape(beta_reversal.shape + (1,))
        beta_momentum_z = _softplus_inv(beta_momentum).reshape(beta_momentum.shape + (1,))
        sigma_z = _softplus_inv(sigma).reshape(sigma.shape + (1,))
        eta_z = _logit(eta).reshape(eta.shape + (1,))
        sigma_extra_z = _softplus_inv(sigma_extra).reshape(sigma_extra.shape + (1,))
        nu_z = _softplus_inv(nu - NU_BURST_FLOOR).reshape(nu.shape + (1,))
        vol_z = np.concatenate(
            [
                alpha_z,
                phi_z,
                beta_reversal_z,
                beta_momentum_z,
                sigma_z,
                eta_z,
                sigma_extra_z,
                nu_z,
            ],
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
        beta_reversal_z = vol_z[..., 2]
        beta_momentum_z = vol_z[..., 3]
        sigma_z = vol_z[..., 4]
        eta_z = vol_z[..., 5]
        sigma_extra_z = vol_z[..., 6]
        nu_z = vol_z[..., 7]

        alpha_const = 0.5 * math.log(2.0 * math.pi) + math.log(self.alpha_v_sd)
        phi_const = 0.5 * math.log(2.0 * math.pi) + math.log(self.phi_v_unconstrained_sd)
        alpha_lp = -0.5 * (alpha_z / self.alpha_v_sd) ** 2 - alpha_const
        phi_lp = -0.5 * (phi_z / self.phi_v_unconstrained_sd) ** 2 - phi_const

        beta_reversal = _softplus(beta_reversal_z)
        beta_momentum = _softplus(beta_momentum_z)
        sigma = _softplus(sigma_z)
        eta = _sigmoid(eta_z)
        sigma_extra = _softplus(sigma_extra_z)
        nu_shift = _softplus(nu_z)

        halfnormal_const = lambda sd: 0.5 * math.log(2.0 / math.pi) - math.log(sd)
        beta_reversal_lp = (
            halfnormal_const(self.beta_reversal_sd)
            - 0.5 * (beta_reversal / self.beta_reversal_sd) ** 2
        )
        beta_momentum_lp = (
            halfnormal_const(self.beta_momentum_sd)
            - 0.5 * (beta_momentum / self.beta_momentum_sd) ** 2
        )
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
            + beta_reversal_lp
            + beta_momentum_lp
            + sigma_lp
            + eta_lp
            + sigma_extra_lp
            + nu_lp
            + _log_softplus_jac(beta_reversal_z)
            + _log_softplus_jac(beta_momentum_z)
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
            self.beta_reversal_sd,
            self.beta_momentum_sd,
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
            self.beta_reversal_sd,
            self.beta_momentum_sd,
            self.sigma_v_sd,
            self.eta_burst_alpha,
            self.eta_burst_beta,
            self.sigma_burst_extra_sd,
            self.nu_burst_gamma_shape,
            self.nu_burst_gamma_scale,
        )
