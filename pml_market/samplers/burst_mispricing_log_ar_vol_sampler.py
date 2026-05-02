"""Forward sampler for `BurstMispricingLogARVolModel`."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ._base import (
    DEFAULT_ETA_BURST,
    DEFAULT_NU_BURST,
    DEFAULT_SIGMA_BURST_EXTRA,
    JointMarkovSamplerBase,
    draw_nonnegative_normal_u,
    draw_nonnegative_student_u,
    reversal_pressure,
    scalar,
)
from .mispricing_log_ar_vol_sampler import default_mispricing_theta


def default_burst_mispricing_theta() -> dict[str, np.ndarray]:
    """Default synthetic theta for `BurstMispricingLogARVolModel`."""
    theta = default_mispricing_theta()
    theta.update({
        "eta_burst": np.array(DEFAULT_ETA_BURST),
        "sigma_burst_extra": np.array(DEFAULT_SIGMA_BURST_EXTRA),
        "nu_burst": np.array(DEFAULT_NU_BURST),
    })
    return theta


class BurstMispricingLogARVolSampler(JointMarkovSamplerBase):
    """Sampler for `BurstMispricingLogARVolModel`."""

    def default_theta(self) -> dict[str, np.ndarray]:
        return default_burst_mispricing_theta()

    def _next_u(self, u_prev: float, dx_prev: float, y: int,
                theta: Mapping[str, object], rng: np.random.Generator) -> float:
        alpha = scalar(theta, "alpha_v")
        phi = scalar(theta, "phi_v")
        beta = scalar(theta, "beta_misprice")
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

        mean = base_mean + beta * reversal_pressure(dx_prev, y)
        return draw_nonnegative_normal_u(mean, sigma, rng, self.max_resample_attempts)
