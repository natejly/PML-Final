"""Prior for the volume-gated reversal/momentum burst log-volume AR model.

This extends `ReversalMomentumBurstLogARVolPrior` with two interpretable gate
parameters:

    volume_gate_center   in R      threshold in u = log(1 + v) space
    volume_gate_slope    > 0      sharpness of the low/high-volume split

The inherited beta terms remain nonnegative.  The gate makes reversal strongest
in low-volume regimes and momentum strongest in high-volume regimes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping

import numpy as np

from ..core import Prior
from .base_prior import BasePrior
from .burst_mispricing_log_ar_vol_prior import (
    _halfnormal_logpdf_np,
    _log_softplus_jac,
    _normal_logpdf_np,
    _softplus,
    _softplus_inv,
)
from .reversal_momentum_burst_log_ar_vol_prior import (
    ReversalMomentumBurstLogARVolPrior,
    log_prior as _parent_log_prior,
    log_prior_batched as _parent_log_prior_batched,
)


VOLUME_GATE_CENTER_PRIOR_SD = 5.0
VOLUME_GATE_SLOPE_PRIOR_SD = 2.0


class GatedReversalMomentumBurstLogARVolPrior(Prior):
    """Prior + bijector for `GatedReversalMomentumBurstLogARVolModel`."""

    PARAM_NAMES = tuple(ReversalMomentumBurstLogARVolPrior.PARAM_NAMES) + (
        "volume_gate_center",
        "volume_gate_slope",
    )

    def __init__(
        self,
        base_prior: BasePrior | None = None,
        volume_gate_center_sd: float = VOLUME_GATE_CENTER_PRIOR_SD,
        volume_gate_slope_sd: float = VOLUME_GATE_SLOPE_PRIOR_SD,
        **parent_kwargs: Any,
    ):
        self.parent = ReversalMomentumBurstLogARVolPrior(
            base_prior=base_prior,
            **parent_kwargs,
        )
        self.base = self.parent.base
        self.volume_gate_center_sd = float(volume_gate_center_sd)
        self.volume_gate_slope_sd = float(volume_gate_slope_sd)
        self.PARENT_UNCONSTRAINED_DIM = int(self.parent.UNCONSTRAINED_DIM)
        self.UNCONSTRAINED_DIM = self.PARENT_UNCONSTRAINED_DIM + 2

    def sample(self, rng: np.random.Generator, n: int = 1) -> Dict[str, np.ndarray]:
        n = int(n)
        theta = self.parent.sample(rng, n)
        theta["volume_gate_center"] = rng.normal(
            0.0,
            self.volume_gate_center_sd,
            size=n,
        ).astype(np.float64)
        theta["volume_gate_slope"] = np.abs(
            rng.normal(0.0, self.volume_gate_slope_sd, size=n)
        ).astype(np.float64)
        return theta

    def transform(self, z):
        """Map unconstrained z (..., d+2) -> theta dict + log|det dtheta/dz|."""
        parent_z = z[..., :self.PARENT_UNCONSTRAINED_DIM]
        gate_z = z[..., self.PARENT_UNCONSTRAINED_DIM:]

        theta, log_jac = self.parent.transform(parent_z)
        center_z = gate_z[..., 0]
        slope_z = gate_z[..., 1]
        theta["volume_gate_center"] = center_z
        theta["volume_gate_slope"] = _softplus(slope_z)
        return theta, log_jac + _log_softplus_jac(slope_z)

    def to_unconstrained(self, theta: Mapping[str, Any]) -> np.ndarray:
        parent_z = self.parent.to_unconstrained(theta)
        center = np.asarray(theta["volume_gate_center"], dtype=np.float64)
        slope = np.asarray(theta["volume_gate_slope"], dtype=np.float64)

        center_z = center.reshape(center.shape + (1,))
        slope_z = _softplus_inv(slope).reshape(slope.shape + (1,))
        gate_z = np.concatenate([center_z, slope_z], axis=-1)
        return np.concatenate([parent_z, gate_z], axis=-1)

    def log_prior_unconstrained(self, z):
        """log pi(z) = log Pi(theta(z)) + log|det dtheta/dz|."""
        parent_z = z[..., :self.PARENT_UNCONSTRAINED_DIM]
        gate_z = z[..., self.PARENT_UNCONSTRAINED_DIM:]
        parent_lp = self.parent.log_prior_unconstrained(parent_z)

        center_z = gate_z[..., 0]
        slope_z = gate_z[..., 1]
        slope = _softplus(slope_z)

        center_const = 0.5 * math.log(2.0 * math.pi) + math.log(self.volume_gate_center_sd)
        center_lp = -0.5 * (center_z / self.volume_gate_center_sd) ** 2 - center_const
        slope_lp = (
            0.5 * math.log(2.0 / math.pi)
            - math.log(self.volume_gate_slope_sd)
            - 0.5 * (slope / self.volume_gate_slope_sd) ** 2
        )
        return parent_lp + center_lp + slope_lp + _log_softplus_jac(slope_z)

    def log_prior(self, theta: Mapping[str, Any]) -> float:
        lp = _parent_log_prior(
            theta,
            self.parent.base,
            self.parent.alpha_v_sd,
            self.parent.phi_v_unconstrained_sd,
            self.parent.beta_reversal_sd,
            self.parent.beta_momentum_sd,
            self.parent.sigma_v_sd,
            self.parent.eta_burst_alpha,
            self.parent.eta_burst_beta,
            self.parent.sigma_burst_extra_sd,
            self.parent.nu_burst_gamma_shape,
            self.parent.nu_burst_gamma_scale,
        )
        center = float(np.asarray(theta["volume_gate_center"]))
        slope = float(np.asarray(theta["volume_gate_slope"]))
        lp += float(_normal_logpdf_np(center, 0.0, self.volume_gate_center_sd))
        lp += float(_halfnormal_logpdf_np(slope, self.volume_gate_slope_sd))
        return lp

    def log_prior_batched(self, theta: Mapping[str, np.ndarray]) -> np.ndarray:
        out = _parent_log_prior_batched(
            theta,
            self.parent.base,
            self.parent.alpha_v_sd,
            self.parent.phi_v_unconstrained_sd,
            self.parent.beta_reversal_sd,
            self.parent.beta_momentum_sd,
            self.parent.sigma_v_sd,
            self.parent.eta_burst_alpha,
            self.parent.eta_burst_beta,
            self.parent.sigma_burst_extra_sd,
            self.parent.nu_burst_gamma_shape,
            self.parent.nu_burst_gamma_scale,
        )
        center = np.asarray(theta["volume_gate_center"], dtype=np.float64)
        slope = np.asarray(theta["volume_gate_slope"], dtype=np.float64)
        out = out + _normal_logpdf_np(center, 0.0, self.volume_gate_center_sd)
        out = out + _halfnormal_logpdf_np(slope, self.volume_gate_slope_sd)
        return out
