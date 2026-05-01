"""Robust joint Markov model with Y-independent volume bursts.

This model preserves the paper's base increment law,

    f_y(dx_t | v_t, theta_x),

and adds a two-channel Markov law for transformed volume u_t = log(1 + v_t).
Most periods use the outcome-oriented mispricing AR channel,

    u_t ~ Normal(
        alpha_v + phi_v * (u_{t-1} - alpha_v)
        + beta_misprice * [-(2y - 1) dx_{t-1}]_+,
        sigma_v^2
    ),

but a period can instead be explained by a Y-independent Student-t burst:

    u_t ~ t_nu_burst(
        alpha_v + phi_v * (u_{t-1} - alpha_v),
        (sigma_v + sigma_burst_extra)^2
    ).

The burst channel is the model's "not every volume spike is evidence about the
outcome" escape hatch: large generic news/attention spikes need not be routed
through the mispricing term.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from scipy.special import gammaln as _np_lgamma

from ..core import Model
from . import base_model


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


def _log(x):
    return _backend(x).log(x)


def _lgamma(x):
    if _is_torch(x):
        import torch

        return torch.lgamma(x)
    return _np_lgamma(x)


def _as_backend_array(x, like):
    if _is_torch(like):
        import torch

        return torch.as_tensor(x, dtype=like.dtype, device=like.device)
    return np.asarray(x, dtype=np.float64)


def _positive_part(x):
    if _is_torch(x):
        return x.clamp(min=0.0)
    return np.maximum(x, 0.0)


def _clamp_prob(p):
    if _is_torch(p):
        return p.clamp(min=1e-12, max=1.0 - 1e-12)
    return np.clip(p, 1e-12, 1.0 - 1e-12)


def _logaddexp(a, b):
    if _is_torch(a) or _is_torch(b):
        import torch

        return torch.logaddexp(a, b)
    return np.logaddexp(a, b)


def _expand_time_param(x, ref):
    """Add a trailing time axis to batched parameters when needed."""
    if hasattr(x, "ndim") and hasattr(ref, "ndim") and x.ndim > 0 and ref.ndim > x.ndim:
        return x[..., None]
    return x


_LOG_2PI = math.log(2.0 * math.pi)
_SCALE_FLOOR = 1e-8


def _floor_scale(scale):
    if _is_torch(scale):
        return scale.clamp(min=_SCALE_FLOOR)
    return np.maximum(scale, _SCALE_FLOOR)


def _normal_logpdf(x, mean, scale):
    scale = _floor_scale(scale)
    z = (x - mean) / scale
    return -0.5 * (z * z + _LOG_2PI) - _log(scale)


def _student_t_logpdf(x, mean, scale, nu):
    scale = _floor_scale(scale)
    z = (x - mean) / scale
    half_nu = 0.5 * nu
    log_norm = (
        _lgamma(half_nu + 0.5)
        - _lgamma(half_nu)
        - 0.5 * _log(nu * math.pi)
    )
    z2 = z * z
    log1p_zsq_over_nu = _log(nu + z2) - _log(nu)
    return log_norm - 0.5 * (nu + 1.0) * log1p_zsq_over_nu - _log(scale)


def mispricing_pressure(dx, y: int):
    """Return [-(2y - 1) dx]_+, positive after a move away from truth."""
    sign = 2.0 * int(y) - 1.0
    return _positive_part(-sign * dx)


def _volume_logpdf_from_terms(u_now, u_prev, dx_prev, y: int, theta_v: Mapping[str, Any]):
    alpha = _expand_time_param(theta_v["alpha_v"], u_prev)
    phi = _expand_time_param(theta_v["phi_v"], u_prev)
    beta = _expand_time_param(theta_v["beta_misprice"], u_prev)
    sigma = _expand_time_param(theta_v["sigma_v"], u_prev)
    eta = _clamp_prob(_expand_time_param(theta_v["eta_burst"], u_prev))
    sigma_extra = _expand_time_param(theta_v["sigma_burst_extra"], u_prev)
    sigma_burst = sigma + sigma_extra
    nu_burst = _expand_time_param(theta_v["nu_burst"], u_prev)

    base_mean = alpha + phi * (u_prev - alpha)
    pressure = mispricing_pressure(dx_prev, y)
    mispricing_mean = base_mean + beta * pressure

    normal_logp = _normal_logpdf(u_now, mispricing_mean, sigma)
    burst_logp = _student_t_logpdf(u_now, base_mean, sigma_burst, nu_burst)
    return _logaddexp(_log(1.0 - eta) + normal_logp, _log(eta) + burst_logp)


def log_volume_transition_logpdf(dx, v, y: int, theta_v: Mapping[str, Any]):
    """Per-step robust log p(v_t | dx_{t-1}, v_{t-1}, y, theta_v).

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
    """Total robust log-volume likelihood."""
    return log_volume_transition_logpdf(dx, v, y, theta_v).sum(axis=-1)


def joint_per_time_logpdf(dx, v, y: int, theta_x: Mapping[str, Any], theta_v: Mapping[str, Any]):
    """Vectorized per-time joint factor for p(dx, v | y, theta)."""
    log_dx = base_model.mixture_logpdf(dx, v, y, theta_x)
    log_v = log_volume_transition_logpdf(dx, v, y, theta_v)
    return log_dx + log_v


def joint_loglik(dx, v, y: int, theta_x: Mapping[str, Any], theta_v: Mapping[str, Any]):
    """Total joint log-likelihood for (dx, v) under the robust Markov model."""
    return joint_per_time_logpdf(dx, v, y, theta_x, theta_v).sum(axis=-1)


class BurstMispricingLogARVolModel(Model):
    """Base increments plus mispricing volume with Y-independent burst spikes."""

    PARAM_NAMES = tuple(base_model.PARAM_NAMES) + (
        "alpha_v",
        "phi_v",
        "beta_misprice",
        "sigma_v",
        "eta_burst",
        "sigma_burst_extra",
        "nu_burst",
    )

    mispricing_pressure = staticmethod(mispricing_pressure)
    log_volume_transition_logpdf = staticmethod(log_volume_transition_logpdf)
    log_volume_loglik = staticmethod(log_volume_loglik)
    joint_per_time_logpdf = staticmethod(joint_per_time_logpdf)
    joint_loglik = staticmethod(joint_loglik)

    def mixture_logpdf(self, dx, v, y: int, theta: Mapping[str, Any]):
        """Vectorized helper returning per-time joint log factors."""
        theta_x = {k: theta[k] for k in base_model.PARAM_NAMES}
        theta_v = {
            k: theta[k]
            for k in (
                "alpha_v",
                "phi_v",
                "beta_misprice",
                "sigma_v",
                "eta_burst",
                "sigma_burst_extra",
                "nu_burst",
            )
        }
        return joint_per_time_logpdf(dx, v, y, theta_x, theta_v)

    def incremental_logpdf(self, dx, v, y: int, theta: Mapping[str, Any], t: int):
        """One-step joint factor for SMC."""
        theta_x = {k: theta[k] for k in base_model.PARAM_NAMES}
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
        theta_v = {
            k: theta[k]
            for k in (
                "alpha_v",
                "phi_v",
                "beta_misprice",
                "sigma_v",
                "eta_burst",
                "sigma_burst_extra",
                "nu_burst",
            )
        }
        log_v = _volume_logpdf_from_terms(u[t], u[t - 1], dx_arr[t - 1], y, theta_v) - u[t]
        return log_dx + log_v

    def loglik(self, dx, v, y: int, theta: Mapping[str, Any]):
        theta_x = {k: theta[k] for k in base_model.PARAM_NAMES}
        theta_v = {
            k: theta[k]
            for k in (
                "alpha_v",
                "phi_v",
                "beta_misprice",
                "sigma_v",
                "eta_burst",
                "sigma_burst_extra",
                "nu_burst",
            )
        }
        return joint_loglik(dx, v, y, theta_x, theta_v)
