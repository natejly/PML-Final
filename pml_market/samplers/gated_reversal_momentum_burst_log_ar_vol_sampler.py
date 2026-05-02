"""Forward sampler for `GatedReversalMomentumBurstLogARVolModel`."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ._base import (
    DEFAULT_ALPHA_V,
    DEFAULT_BETA_MOMENTUM,
    DEFAULT_BETA_REVERSAL,
    DEFAULT_ETA_BURST,
    DEFAULT_NU_BURST,
    DEFAULT_PHI_V,
    DEFAULT_SIGMA_BURST_EXTRA,
    DEFAULT_SIGMA_V,
    JointMarkovSamplerBase,
    base_theta,
    draw_nonnegative_normal_u,
    draw_nonnegative_student_u,
    momentum_pressure,
    reversal_pressure,
    scalar,
)


DEFAULT_VOLUME_GATE_CENTER = DEFAULT_ALPHA_V
DEFAULT_VOLUME_GATE_SLOPE = 3.0


def default_gated_reversal_momentum_burst_theta() -> dict[str, np.ndarray]:
    """Default synthetic theta for `GatedReversalMomentumBurstLogARVolModel`."""
    theta = base_theta()
    theta.update({
        "alpha_v": np.array(DEFAULT_ALPHA_V),
        "phi_v": np.array(DEFAULT_PHI_V),
        "beta_reversal": np.array(DEFAULT_BETA_REVERSAL),
        "beta_momentum": np.array(DEFAULT_BETA_MOMENTUM),
        "volume_gate_center": np.array(DEFAULT_VOLUME_GATE_CENTER),
        "volume_gate_slope": np.array(DEFAULT_VOLUME_GATE_SLOPE),
        "sigma_v": np.array(DEFAULT_SIGMA_V),
        "eta_burst": np.array(DEFAULT_ETA_BURST),
        "sigma_burst_extra": np.array(DEFAULT_SIGMA_BURST_EXTRA),
        "nu_burst": np.array(DEFAULT_NU_BURST),
    })
    return theta


def high_volume_gate(u_prev: float, center: float, slope: float) -> float:
    z = float(slope) * (float(u_prev) - float(center))
    if z >= 0.0:
        return float(1.0 / (1.0 + np.exp(-z)))
    ez = float(np.exp(z))
    return float(ez / (1.0 + ez))


class GatedReversalMomentumBurstLogARVolSampler(JointMarkovSamplerBase):
    """Sampler for `GatedReversalMomentumBurstLogARVolModel`."""

    def default_theta(self) -> dict[str, np.ndarray]:
        return default_gated_reversal_momentum_burst_theta()

    def _next_u(self, u_prev: float, dx_prev: float, y: int,
                theta: Mapping[str, object], rng: np.random.Generator) -> float:
        alpha = scalar(theta, "alpha_v")
        phi = scalar(theta, "phi_v")
        beta_reversal = scalar(theta, "beta_reversal")
        beta_momentum = scalar(theta, "beta_momentum")
        center = scalar(theta, "volume_gate_center")
        slope = scalar(theta, "volume_gate_slope")
        sigma = scalar(theta, "sigma_v")
        eta = np.clip(scalar(theta, "eta_burst"), 0.0, 1.0)
        sigma_burst = sigma + scalar(theta, "sigma_burst_extra")
        nu_burst = scalar(theta, "nu_burst")

        base_mean = alpha + phi * (u_prev - alpha)
        if rng.uniform() < eta:
            return draw_nonnegative_student_u(
                base_mean,
                sigma_burst,
                nu_burst,
                rng,
                self.max_resample_attempts,
            )

        high = high_volume_gate(u_prev, center, slope)
        low = 1.0 - high
        mean = (
            base_mean
            + beta_reversal * reversal_pressure(dx_prev, y) * low
            + beta_momentum * momentum_pressure(dx_prev, y) * high
        )
        return draw_nonnegative_normal_u(mean, sigma, rng, self.max_resample_attempts)
