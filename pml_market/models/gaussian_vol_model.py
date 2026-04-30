"""Markovian volume extension: Gaussian random-walk volume dynamics.

This module keeps the existing price-increment model from `base_model.py` and adds
an explicit Markov factor for volume:

    v_t | v_{t-1}, sigma_v ~ Normal(v_{t-1}, sigma_v^2),  t >= 2.

By default, t=1 contributes no volume likelihood term (constant wrt parameters),
which is often convenient for sequential inference. Optionally, an initial
Normal prior term for v_1 can be included.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

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


def volume_transition_logpdf(
    v,
    theta_v: Mapping[str, Any],
    include_initial: bool = False,
    v1_mean: float = 0.0,
    v1_scale: float = 10.0,
):
    """Per-step Markov volume log-density.

    Parameters
    ----------
    v : (T,) array-like volume sequence.
    theta_v : dict containing key `sigma_v`.
    include_initial : if True, adds log p(v_1) as Normal(v1_mean, v1_scale^2).
                      if False, first term is 0.0 (constant wrt theta_v).

    Returns
    -------
    per_step_logp : (..., T) array.
        Supports broadcasting over leading batch dims of theta_v["sigma_v"].
    """
    sigma_v = theta_v["sigma_v"]

    # Build an explicit output with shape batch_shape + (T,), where batch_shape
    # comes from sigma_v (e.g., N particles) and T is the time dimension of v.
    if _is_torch(sigma_v):
        import torch

        v_arr = torch.as_tensor(v, dtype=sigma_v.dtype, device=sigma_v.device)
        T = int(v_arr.shape[-1])
        batch_shape = tuple(sigma_v.shape) if sigma_v.ndim > 0 else ()
        out = torch.zeros(batch_shape + (T,), dtype=sigma_v.dtype, device=sigma_v.device)

        if include_initial:
            out[..., 0] = _normal_logpdf(v_arr[0], v1_mean, v1_scale)

        if T >= 2:
            lead = sigma_v.ndim if sigma_v.ndim > 0 else 0
            v_b = v_arr[(None,) * lead + (slice(None),)]
            sig_b = sigma_v[..., None] if sigma_v.ndim > 0 else sigma_v
            out[..., 1:] = _normal_logpdf(v_b[..., 1:], v_b[..., :-1], sig_b)

        return out

    v_arr = np.asarray(v, dtype=np.float64)
    T = int(v_arr.shape[-1])
    sigma_np = np.asarray(sigma_v, dtype=np.float64)
    batch_shape = tuple(sigma_np.shape) if sigma_np.ndim > 0 else ()
    out = np.zeros(batch_shape + (T,), dtype=np.float64)

    if include_initial:
        out[..., 0] = _normal_logpdf(v_arr[0], v1_mean, v1_scale)

    if T >= 2:
        lead = sigma_np.ndim if sigma_np.ndim > 0 else 0
        v_b = v_arr[(None,) * lead + (slice(None),)]
        sig_b = sigma_np[..., None] if sigma_np.ndim > 0 else sigma_np
        out[..., 1:] = _normal_logpdf(v_b[..., 1:], v_b[..., :-1], sig_b)

    return out


def volume_loglik(
    v,
    theta_v: Mapping[str, Any],
    include_initial: bool = False,
    v1_mean: float = 0.0,
    v1_scale: float = 10.0,
):
    """Total volume log-likelihood under Markov transition model."""
    return volume_transition_logpdf(
        v,
        theta_v,
        include_initial=include_initial,
        v1_mean=v1_mean,
        v1_scale=v1_scale,
    ).sum(axis=-1)


def joint_per_step_logpdf(
    dx,
    v,
    y: int,
    theta_x: Mapping[str, Any],
    theta_v: Mapping[str, Any],
    include_initial: bool = False,
    v1_mean: float = 0.0,
    v1_scale: float = 10.0,
):
    """Per-step log factor for p(dx_t, v_t | history, y, theta_x, theta_v).

    This combines:
      - base price-increment factor from `base_model.py`:
            p(dx_t | v_t, y, theta_x)
      - volume Markov factor:
            p(v_t | v_{t-1}, sigma_v)
    """
    log_dx = base_model.mixture_logpdf(dx, v, y, theta_x)
    log_v = volume_transition_logpdf(
        v,
        theta_v,
        include_initial=include_initial,
        v1_mean=v1_mean,
        v1_scale=v1_scale,
    )
    return log_dx + log_v


def joint_loglik(
    dx,
    v,
    y: int,
    theta_x: Mapping[str, Any],
    theta_v: Mapping[str, Any],
    include_initial: bool = False,
    v1_mean: float = 0.0,
    v1_scale: float = 10.0,
):
    """Total joint log-likelihood for (dx, v) with Markovian volume."""
    return joint_per_step_logpdf(
        dx,
        v,
        y,
        theta_x,
        theta_v,
        include_initial=include_initial,
        v1_mean=v1_mean,
        v1_scale=v1_scale,
    ).sum(axis=-1)


def sample_volumes_markov(
    T: int,
    theta_v: Mapping[str, float],
    v0: float = 1.0,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    clip_min: Optional[float] = 0.0,
) -> np.ndarray:
    """Generate a volume path from the Gaussian Markov transition model.

    v[0] = v0
    v[t] = v[t-1] + sigma_v * eps_t, eps_t ~ N(0, 1)

    `clip_min` keeps volumes nonnegative by clipping after each transition.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    sigma_v = float(np.asarray(theta_v["sigma_v"]))
    out = np.empty(T, dtype=np.float64)
    out[0] = float(v0)
    for t in range(1, T):
        out[t] = out[t - 1] + sigma_v * float(rng.normal())
        if clip_min is not None:
            out[t] = max(float(clip_min), out[t])
    return out


class GaussianVolModel(Model):
    """Joint model with base price increments + Gaussian Markov volume bumps.

    Volume condition implemented here:
      v_t | v_{t-1}, sigma_v ~ Normal(v_{t-1}, sigma_v^2)
    """

    PARAM_NAMES = tuple(base_model.PARAM_NAMES) + ("sigma_v",)

    volume_transition_logpdf = staticmethod(volume_transition_logpdf)
    volume_loglik = staticmethod(volume_loglik)
    joint_per_step_logpdf = staticmethod(joint_per_step_logpdf)
    joint_loglik = staticmethod(joint_loglik)
    sample_volumes_markov = staticmethod(sample_volumes_markov)

    def mixture_logpdf(self, dx, v, y: int, theta: Mapping[str, Any]):
        theta_x = {k: theta[k] for k in base_model.PARAM_NAMES}
        theta_v = {"sigma_v": theta["sigma_v"]}
        return joint_per_step_logpdf(dx, v, y, theta_x, theta_v)

    def loglik(self, dx, v, y: int, theta: Mapping[str, Any]):
        theta_x = {k: theta[k] for k in base_model.PARAM_NAMES}
        theta_v = {"sigma_v": theta["sigma_v"]}
        return joint_loglik(dx, v, y, theta_x, theta_v)
