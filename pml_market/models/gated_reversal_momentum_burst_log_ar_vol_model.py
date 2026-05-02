"""Joint Markov model with volume-gated reversal and momentum channels.

This model preserves the paper's base increment law,

    f_y(dx_t | v_t, theta_x),

and adds an interpretable Markov law for transformed volume u_t = log(1 + v_t).
Most periods use a directional AR channel

    u_t ~ Normal(
        alpha_v + phi_v * (u_{t-1} - alpha_v)
        + beta_reversal * [-(2y - 1) dx_{t-1}]_+ * g_low(u_{t-1})
        + beta_momentum * [ (2y - 1) dx_{t-1}]_+ * g_high(u_{t-1}),
        sigma_v^2
    ),

where

    g_high(u) = sigmoid(volume_gate_slope * (u - volume_gate_center)),
    g_low(u) = 1 - g_high(u).

The economic story is deliberately asymmetric: low-volume wrong-way moves look
like correctable mispricing, while high-volume right-way moves look like visible
information/attention cascades.  A separate Y-independent Student-t burst
channel remains available for generic news or liquidity spikes.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..core import Model
from . import base_model
from .burst_mispricing_log_ar_vol_model import (
    _as_backend_array,
    _clamp_prob,
    _expand_time_param,
    _is_torch,
    _log,
    _logaddexp,
    _normal_logpdf,
    _positive_part,
    _student_t_logpdf,
)


VOLUME_PARAM_NAMES = (
    "alpha_v",
    "phi_v",
    "beta_reversal",
    "beta_momentum",
    "volume_gate_center",
    "volume_gate_slope",
    "sigma_v",
    "eta_burst",
    "sigma_burst_extra",
    "nu_burst",
)


def _sigmoid(x):
    if _is_torch(x):
        import torch

        return torch.sigmoid(x)
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def reversal_pressure(dx, y: int):
    """Return [-(2y - 1) dx]_+, positive after a move away from truth."""
    sign = 2.0 * int(y) - 1.0
    return _positive_part(-sign * dx)


def momentum_pressure(dx, y: int):
    """Return [(2y - 1) dx]_+, positive after a move toward truth."""
    sign = 2.0 * int(y) - 1.0
    return _positive_part(sign * dx)


def high_volume_gate(u_prev, theta_v: Mapping[str, Any]):
    """Soft high-volume gate in transformed-volume space."""
    center = _expand_time_param(theta_v["volume_gate_center"], u_prev)
    slope = _expand_time_param(theta_v["volume_gate_slope"], u_prev)
    return _sigmoid(slope * (u_prev - center))


def low_volume_gate(u_prev, theta_v: Mapping[str, Any]):
    """Complementary low-volume gate."""
    return 1.0 - high_volume_gate(u_prev, theta_v)


def _volume_logpdf_from_terms(u_now, u_prev, dx_prev, y: int, theta_v: Mapping[str, Any]):
    alpha = _expand_time_param(theta_v["alpha_v"], u_prev)
    phi = _expand_time_param(theta_v["phi_v"], u_prev)
    beta_reversal = _expand_time_param(theta_v["beta_reversal"], u_prev)
    beta_momentum = _expand_time_param(theta_v["beta_momentum"], u_prev)
    sigma = _expand_time_param(theta_v["sigma_v"], u_prev)
    eta = _clamp_prob(_expand_time_param(theta_v["eta_burst"], u_prev))
    sigma_extra = _expand_time_param(theta_v["sigma_burst_extra"], u_prev)
    sigma_burst = sigma + sigma_extra
    nu_burst = _expand_time_param(theta_v["nu_burst"], u_prev)

    base_mean = alpha + phi * (u_prev - alpha)
    oriented_mean = (
        base_mean
        + beta_reversal * reversal_pressure(dx_prev, y) * low_volume_gate(u_prev, theta_v)
        + beta_momentum * momentum_pressure(dx_prev, y) * high_volume_gate(u_prev, theta_v)
    )

    normal_logp = _normal_logpdf(u_now, oriented_mean, sigma)
    burst_logp = _student_t_logpdf(u_now, base_mean, sigma_burst, nu_burst)
    return _logaddexp(_log(1.0 - eta) + normal_logp, _log(eta) + burst_logp)


def log_volume_transition_logpdf(dx, v, y: int, theta_v: Mapping[str, Any]):
    """Per-step log p(v_t | dx_{t-1}, v_{t-1}, y, theta_v).

    Returns shape (..., T). The first term is zero by convention because no
    previous history slice is available for the first observed increment.
    """
    sigma = theta_v["sigma_v"]

    if _is_torch(sigma):
        import torch

        dx_arr = torch.as_tensor(dx, dtype=sigma.dtype, device=sigma.device)
        v_arr = torch.as_tensor(v, dtype=sigma.dtype, device=sigma.device)
        u = torch.log1p(v_arr)
        T = int(u.shape[-1])
        batch_shape = tuple(sigma.shape) if sigma.ndim > 0 else ()
        out = torch.zeros(batch_shape + (T,), dtype=sigma.dtype, device=sigma.device)

        if T >= 2:
            lead = sigma.ndim if sigma.ndim > 0 else 0
            u_b = u[(None,) * lead + (slice(None),)]
            dx_b = dx_arr[(None,) * lead + (slice(None),)]
            out[..., 1:] = (
                _volume_logpdf_from_terms(u_b[..., 1:], u_b[..., :-1], dx_b[..., :-1], y, theta_v)
                - u_b[..., 1:]
            )

        return out

    dx_arr = np.asarray(dx, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    u = np.log1p(v_arr)
    T = int(u.shape[-1])
    sigma_np = np.asarray(sigma, dtype=np.float64)
    batch_shape = tuple(sigma_np.shape) if sigma_np.ndim > 0 else ()
    out = np.zeros(batch_shape + (T,), dtype=np.float64)

    if T >= 2:
        lead = sigma_np.ndim if sigma_np.ndim > 0 else 0
        u_b = u[(None,) * lead + (slice(None),)]
        dx_b = dx_arr[(None,) * lead + (slice(None),)]
        out[..., 1:] = (
            _volume_logpdf_from_terms(u_b[..., 1:], u_b[..., :-1], dx_b[..., :-1], y, theta_v)
            - u_b[..., 1:]
        )

    return out


def log_volume_loglik(dx, v, y: int, theta_v: Mapping[str, Any]):
    """Total log-volume likelihood."""
    return log_volume_transition_logpdf(dx, v, y, theta_v).sum(axis=-1)


def joint_per_time_logpdf(dx, v, y: int, theta_x: Mapping[str, Any], theta_v: Mapping[str, Any]):
    """Vectorized per-time joint factor for p(dx, v | y, theta)."""
    log_dx = base_model.mixture_logpdf(dx, v, y, theta_x)
    log_v = log_volume_transition_logpdf(dx, v, y, theta_v)
    return log_dx + log_v


def joint_loglik(dx, v, y: int, theta_x: Mapping[str, Any], theta_v: Mapping[str, Any]):
    """Total joint log-likelihood for (dx, v)."""
    return joint_per_time_logpdf(dx, v, y, theta_x, theta_v).sum(axis=-1)


class GatedReversalMomentumBurstLogARVolModel(Model):
    """Base increments plus volume-gated reversal, momentum, and burst volume."""

    PARAM_NAMES = tuple(base_model.PARAM_NAMES) + VOLUME_PARAM_NAMES

    reversal_pressure = staticmethod(reversal_pressure)
    momentum_pressure = staticmethod(momentum_pressure)
    high_volume_gate = staticmethod(high_volume_gate)
    low_volume_gate = staticmethod(low_volume_gate)
    log_volume_transition_logpdf = staticmethod(log_volume_transition_logpdf)
    log_volume_loglik = staticmethod(log_volume_loglik)
    joint_per_time_logpdf = staticmethod(joint_per_time_logpdf)
    joint_loglik = staticmethod(joint_loglik)

    def _split_theta(self, theta: Mapping[str, Any]):
        theta_x = {k: theta[k] for k in base_model.PARAM_NAMES}
        theta_v = {k: theta[k] for k in VOLUME_PARAM_NAMES}
        return theta_x, theta_v

    def mixture_logpdf(self, dx, v, y: int, theta: Mapping[str, Any]):
        """Vectorized helper returning per-time joint log factors."""
        theta_x, theta_v = self._split_theta(theta)
        return joint_per_time_logpdf(dx, v, y, theta_x, theta_v)

    def incremental_logpdf(self, dx, v, y: int, theta: Mapping[str, Any], t: int):
        """One-step joint factor for SMC."""
        theta_x, theta_v = self._split_theta(theta)
        sigma = theta["sigma_v"]

        if _is_torch(sigma):
            dx_arr = _as_backend_array(dx, sigma)
            v_arr = _as_backend_array(v, sigma)
            dx_step = dx_arr[t:t + 1]
            v_step = v_arr[t:t + 1]
        else:
            dx_arr = np.asarray(dx, dtype=np.float64)
            v_arr = np.asarray(v, dtype=np.float64)
            dx_step = dx_arr[t:t + 1]
            v_step = v_arr[t:t + 1]

        log_dx = base_model.mixture_logpdf(dx_step, v_step, y, theta_x)[..., 0]

        if t == 0:
            return log_dx

        u = _log(1.0 + v_arr)
        log_v = _volume_logpdf_from_terms(u[t], u[t - 1], dx_arr[t - 1], y, theta_v) - u[t]
        return log_dx + log_v

    def loglik(self, dx, v, y: int, theta: Mapping[str, Any]):
        theta_x, theta_v = self._split_theta(theta)
        return joint_loglik(dx, v, y, theta_x, theta_v)
