"""Joint Markov model with mispricing-reactive log-volume dynamics.

This model preserves the paper's base increment law,

    f_y(dx_t | v_t, theta_x),

and adds a Markov volume law driven by the previous price increment:

    u_t = log(1 + v_t)
    u_t | h_{t-1}, Y=y, theta_v
        ~ Normal(
            alpha_v + phi_v * (u_{t-1} - alpha_v)
            + beta_misprice * [-(2y - 1) dx_{t-1}]_+,
            sigma_v^2
          ).

The nonnegative beta_misprice parameter is an orientation constraint: volume
is allowed to increase after a move away from the eventual truth.  This is the
volume-side analogue of the base model's nonnegative drift magnitudes.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

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


def _as_backend_array(x, like):
    if _is_torch(like):
        import torch

        return torch.as_tensor(x, dtype=like.dtype, device=like.device)
    return np.asarray(x, dtype=np.float64)


def _positive_part(x):
    if _is_torch(x):
        return x.clamp(min=0.0)
    return np.maximum(x, 0.0)


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


def mispricing_pressure(dx, y: int):
    """Return [-(2y - 1) dx]_+, positive after a move away from truth."""
    sign = 2.0 * int(y) - 1.0
    return _positive_part(-sign * dx)


def log_volume_transition_logpdf(dx, v, y: int, theta_v: Mapping[str, Any]):
    """Per-step log p(v_t | dx_{t-1}, v_{t-1}, y, theta_v).

    Returns shape (..., T). The first term is zero by convention because no
    previous history slice is available for the first observed increment.
    """
    alpha = theta_v["alpha_v"]
    phi = theta_v["phi_v"]
    beta = theta_v["beta_misprice"]
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
            alpha_b = alpha[..., None] if getattr(alpha, "ndim", 0) > 0 else alpha
            phi_b = phi[..., None] if getattr(phi, "ndim", 0) > 0 else phi
            beta_b = beta[..., None] if getattr(beta, "ndim", 0) > 0 else beta
            sig_b = sigma[..., None] if sigma.ndim > 0 else sigma
            pressure = mispricing_pressure(dx_b[..., :-1], y)
            mean = alpha_b + phi_b * (u_b[..., :-1] - alpha_b) + beta_b * pressure
            out[..., 1:] = _normal_logpdf(u_b[..., 1:], mean, sig_b) - u_b[..., 1:]

        return out

    dx_arr = np.asarray(dx, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    u = np.log1p(v_arr)
    T = int(u.shape[-1])
    sigma_np = np.asarray(sigma, dtype=np.float64)
    alpha_np = np.asarray(alpha, dtype=np.float64)
    phi_np = np.asarray(phi, dtype=np.float64)
    beta_np = np.asarray(beta, dtype=np.float64)
    batch_shape = tuple(sigma_np.shape) if sigma_np.ndim > 0 else ()
    out = np.zeros(batch_shape + (T,), dtype=np.float64)

    if T >= 2:
        lead = sigma_np.ndim if sigma_np.ndim > 0 else 0
        u_b = u[(None,) * lead + (slice(None),)]
        dx_b = dx_arr[(None,) * lead + (slice(None),)]
        alpha_b = alpha_np[..., None] if alpha_np.ndim > 0 else alpha_np
        phi_b = phi_np[..., None] if phi_np.ndim > 0 else phi_np
        beta_b = beta_np[..., None] if beta_np.ndim > 0 else beta_np
        sig_b = sigma_np[..., None] if sigma_np.ndim > 0 else sigma_np
        pressure = mispricing_pressure(dx_b[..., :-1], y)
        mean = alpha_b + phi_b * (u_b[..., :-1] - alpha_b) + beta_b * pressure
        out[..., 1:] = _normal_logpdf(u_b[..., 1:], mean, sig_b) - u_b[..., 1:]

    return out


def log_volume_loglik(dx, v, y: int, theta_v: Mapping[str, Any]):
    """Total log-volume likelihood under the mispricing log-AR process."""
    return log_volume_transition_logpdf(dx, v, y, theta_v).sum(axis=-1)


def joint_per_time_logpdf(dx, v, y: int, theta_x: Mapping[str, Any], theta_v: Mapping[str, Any]):
    """Vectorized per-time joint factor for p(dx, v | y, theta)."""
    log_dx = base_model.mixture_logpdf(dx, v, y, theta_x)
    log_v = log_volume_transition_logpdf(dx, v, y, theta_v)
    return log_dx + log_v


def joint_loglik(dx, v, y: int, theta_x: Mapping[str, Any], theta_v: Mapping[str, Any]):
    """Total joint log-likelihood for (dx, v) under the joint Markov model."""
    return joint_per_time_logpdf(dx, v, y, theta_x, theta_v).sum(axis=-1)


class MispricingLogARVolModel(Model):
    """Base increments plus mispricing-reactive mean-reverting log-volume."""

    PARAM_NAMES = tuple(base_model.PARAM_NAMES) + (
        "alpha_v",
        "phi_v",
        "beta_misprice",
        "sigma_v",
    )

    mispricing_pressure = staticmethod(mispricing_pressure)
    log_volume_transition_logpdf = staticmethod(log_volume_transition_logpdf)
    log_volume_loglik = staticmethod(log_volume_loglik)
    joint_per_time_logpdf = staticmethod(joint_per_time_logpdf)
    joint_loglik = staticmethod(joint_loglik)

    def mixture_logpdf(self, dx, v, y: int, theta: Mapping[str, Any]):
        """Vectorized helper returning per-time joint log factors."""
        theta_x = {k: theta[k] for k in base_model.PARAM_NAMES}
        theta_v = {k: theta[k] for k in ("alpha_v", "phi_v", "beta_misprice", "sigma_v")}
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
        alpha = theta["alpha_v"]
        phi = theta["phi_v"]
        beta = theta["beta_misprice"]
        pressure = mispricing_pressure(dx_arr[t - 1], y)
        mean = alpha + phi * (u[t - 1] - alpha) + beta * pressure
        log_v = _normal_logpdf(u[t], mean, sigma) - u[t]
        return log_dx + log_v

    def loglik(self, dx, v, y: int, theta: Mapping[str, Any]):
        theta_x = {k: theta[k] for k in base_model.PARAM_NAMES}
        theta_v = {k: theta[k] for k in ("alpha_v", "phi_v", "beta_misprice", "sigma_v")}
        return joint_loglik(dx, v, y, theta_x, theta_v)
