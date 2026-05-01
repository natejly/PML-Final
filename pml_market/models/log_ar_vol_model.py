"""Log-volume AR(1) extension.

This model keeps the base price-increment likelihood unchanged and adds a
mean-reverting Markov model for transformed volume:

    u_t = log(1 + v_t)
    u_t | u_{t-1}, alpha_v, phi_v, sigma_v
        ~ Normal(alpha_v + phi_v * (u_{t-1} - alpha_v), sigma_v^2),
        t >= 2.

By default, t=1 contributes no volume likelihood term.  This is a deliberately
Y-symmetric control model: it improves the geometry of the volume likelihood
relative to a raw-volume random walk, but its exact marginal volume factor
cancels from the Bayes factor when the same prior is used under both outcomes.
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


def log_volume_transition_logpdf(v, theta_v: Mapping[str, Any]):
    """Per-step log p(v_t | v_{t-1}, theta_v) under log-volume AR(1).

    Returns shape (..., T).  The first term is zero by convention because no
    previous volume is available.
    """
    alpha = theta_v["alpha_v"]
    phi = theta_v["phi_v"]
    sigma = theta_v["sigma_v"]

    if _is_torch(sigma):
        import torch

        v_arr = torch.as_tensor(v, dtype=sigma.dtype, device=sigma.device)
        u = torch.log1p(v_arr)
        T = int(u.shape[-1])
        batch_shape = tuple(sigma.shape) if sigma.ndim > 0 else ()
        out = torch.zeros(batch_shape + (T,), dtype=sigma.dtype, device=sigma.device)

        if T >= 2:
            lead = sigma.ndim if sigma.ndim > 0 else 0
            u_b = u[(None,) * lead + (slice(None),)]
            alpha_b = alpha[..., None] if getattr(alpha, "ndim", 0) > 0 else alpha
            phi_b = phi[..., None] if getattr(phi, "ndim", 0) > 0 else phi
            sig_b = sigma[..., None] if sigma.ndim > 0 else sigma
            mean = alpha_b + phi_b * (u_b[..., :-1] - alpha_b)
            out[..., 1:] = _normal_logpdf(u_b[..., 1:], mean, sig_b) - u_b[..., 1:]

        return out

    v_arr = np.asarray(v, dtype=np.float64)
    u = np.log1p(v_arr)
    T = int(u.shape[-1])
    sigma_np = np.asarray(sigma, dtype=np.float64)
    alpha_np = np.asarray(alpha, dtype=np.float64)
    phi_np = np.asarray(phi, dtype=np.float64)
    batch_shape = tuple(sigma_np.shape) if sigma_np.ndim > 0 else ()
    out = np.zeros(batch_shape + (T,), dtype=np.float64)

    if T >= 2:
        lead = sigma_np.ndim if sigma_np.ndim > 0 else 0
        u_b = u[(None,) * lead + (slice(None),)]
        alpha_b = alpha_np[..., None] if alpha_np.ndim > 0 else alpha_np
        phi_b = phi_np[..., None] if phi_np.ndim > 0 else phi_np
        sig_b = sigma_np[..., None] if sigma_np.ndim > 0 else sigma_np
        mean = alpha_b + phi_b * (u_b[..., :-1] - alpha_b)
        out[..., 1:] = _normal_logpdf(u_b[..., 1:], mean, sig_b) - u_b[..., 1:]

    return out


def log_volume_loglik(v, theta_v: Mapping[str, Any]):
    """Total log-volume likelihood under the log-AR process."""
    return log_volume_transition_logpdf(v, theta_v).sum(axis=-1)


def joint_per_time_logpdf(dx, v, y: int, theta_x: Mapping[str, Any], theta_v: Mapping[str, Any]):
    """Vectorized per-time joint factor for p(dx, v | y, theta)."""
    log_dx = base_model.mixture_logpdf(dx, v, y, theta_x)
    log_v = log_volume_transition_logpdf(v, theta_v)
    return log_dx + log_v


def joint_loglik(dx, v, y: int, theta_x: Mapping[str, Any], theta_v: Mapping[str, Any]):
    """Total joint log-likelihood for (dx, v) under the log-AR volume model."""
    return joint_per_time_logpdf(dx, v, y, theta_x, theta_v).sum(axis=-1)


class LogARVolModel(Model):
    """Joint model with base increments plus mean-reverting log-volume AR(1)."""

    PARAM_NAMES = tuple(base_model.PARAM_NAMES) + ("alpha_v", "phi_v", "sigma_v")

    log_volume_transition_logpdf = staticmethod(log_volume_transition_logpdf)
    log_volume_loglik = staticmethod(log_volume_loglik)
    joint_per_time_logpdf = staticmethod(joint_per_time_logpdf)
    joint_loglik = staticmethod(joint_loglik)

    def mixture_logpdf(self, dx, v, y: int, theta: Mapping[str, Any]):
        """Vectorized helper returning per-time joint log factors."""
        theta_x = {k: theta[k] for k in base_model.PARAM_NAMES}
        theta_v = {k: theta[k] for k in ("alpha_v", "phi_v", "sigma_v")}
        return joint_per_time_logpdf(dx, v, y, theta_x, theta_v)

    def incremental_logpdf(self, dx, v, y: int, theta: Mapping[str, Any], t: int):
        """One-step joint factor for SMC."""
        theta_x = {k: theta[k] for k in base_model.PARAM_NAMES}
        log_dx = base_model.mixture_logpdf(
            dx[t:t + 1],
            v[t:t + 1],
            y,
            theta_x,
        )[..., 0]

        if t == 0:
            return log_dx

        u = _log(1.0 + v)
        alpha = theta["alpha_v"]
        phi = theta["phi_v"]
        sigma = theta["sigma_v"]
        mean = alpha + phi * (u[t - 1] - alpha)
        log_v = _normal_logpdf(u[t], mean, sigma) - u[t]
        return log_dx + log_v

    def loglik(self, dx, v, y: int, theta: Mapping[str, Any]):
        theta_x = {k: theta[k] for k in base_model.PARAM_NAMES}
        theta_v = {k: theta[k] for k in ("alpha_v", "phi_v", "sigma_v")}
        return joint_loglik(dx, v, y, theta_x, theta_v)
