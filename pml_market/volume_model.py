"""Volume-aware joint model p(Δx, v | Y, θ).

The base Gaussian latent-type model (`BaseModel`) treats the
trade-volume sequence v_{1:T} as exogenous: it conditions on v but never
puts a probability on it.  This module factors the joint distribution as

    p(Δx_{1:T}, v_{1:T} | Y = y, θ) = p(Δx_{1:T} | v_{1:T}, Y = y, θ_inc)
                                       * p(v_{1:T} | Y = y, θ_vol),

keeping the original increment model intact and adding a parametric
volume model as the second factor.

First-pass volume model (this file): per-bucket lognormal,

    log(1 + v_t)  ~  N(mu_v[Y], sigma_v[Y]^2),  iid across t.

So the four extra parameters are (mu_v[0], mu_v[1], sigma_v[0], sigma_v[1]).
The outcome-dependence on Y is what lets the joint Bayes factor pick up
asymmetries in *trading activity* between Yes-resolving and No-resolving
markets, on top of the increment-direction signal.

The class composes on any inner increment Model, so swapping the latent-type
mixture for a different conditional model is just::

    vol_model = VolumeLognormalModel(increment_model=MyOtherModel())
"""

from __future__ import annotations

import math
from typing import Mapping, Any, Optional

import numpy as np

from .core import Model
from .models.base_model import (
    BaseModel,
    _is_torch, _backend, _log,
)


_LOG_2PI = math.log(2.0 * math.pi)


class VolumeLognormalModel(Model):
    """Joint model with iid lognormal volumes conditional on Y.

    Parameters
    ----------
    increment_model : Model, optional
        Inner conditional model for p(Δx | v, Y, θ_inc).  Defaults to
        `BaseModel()`.

    Notes
    -----
    `mixture_logpdf` returns the per-bucket log of the **joint** density,
    summed by `loglik` to log p(Δx_{1:T}, v_{1:T} | Y = y, θ).  All
    inference routines that consume a `Model` (SMC, VI, diagnostics) work
    unchanged on this joint likelihood; the Bayes factor it produces is
    over the joint observation, picking up the volume signal automatically.
    """

    def __init__(self, increment_model: Optional[Model] = None):
        self.increment_model = increment_model or BaseModel()
        # Param names = inner increment params + the two new volume tensors.
        self.PARAM_NAMES = tuple(self.increment_model.PARAM_NAMES) + ("mu_v", "sigma_v")

    def __repr__(self) -> str:
        return (f"VolumeLognormalModel(increment_model="
                f"{type(self.increment_model).__name__})")

    # ------------------------------------------------------------------
    # Volume term
    # ------------------------------------------------------------------

    @staticmethod
    def _volume_logpdf(v, y: int, theta: Mapping[str, Any]):
        """Per-step log p(v_t | Y=y, θ_vol).

        log(1 + v_t) ~ N(mu_v[y], sigma_v[y]^2), with the |d log(1+v)/dv|
        Jacobian -log(1 + v_t) folded in so the result is a density on v.
        Returns shape (..., T).
        """
        mu_v = theta["mu_v"]                # (..., 2)
        sigma_v = theta["sigma_v"]          # (..., 2)

        # Index outcome y -> (...,) shaped means/stds.
        if _is_torch(mu_v):
            mu_y = mu_v[..., y]
            sig_y = sigma_v[..., y]
        else:
            mu_y = np.asarray(mu_v)[..., y]
            sig_y = np.asarray(sigma_v)[..., y]

        # Broadcast (...,) means with (T,) data -> (..., T).
        log1pv = _log(1.0 + v)              # (T,)
        mu_b = mu_y[..., None]
        sig_b = sig_y[..., None]
        log_sig_b = _log(sig_b)

        z = (log1pv - mu_b) / sig_b
        # Normal log-density of log(1+v_t) + change-of-variables -log(1+v_t).
        return -0.5 * z * z - 0.5 * _LOG_2PI - log_sig_b - log1pv

    # ------------------------------------------------------------------
    # Joint per-step log-density (Model interface)
    # ------------------------------------------------------------------

    def mixture_logpdf(self, dx, v, y: int, theta: Mapping[str, Any]):
        """log p(Δx_t, v_t | v_{<t}, Y=y, θ) for each t.  Shape (..., T).

        Under iid volumes the conditioning on v_{<t} drops out, so this is
        log p(Δx_t | v_t, Y, θ_inc) + log p(v_t | Y, θ_vol).
        """
        inc_theta = {k: theta[k] for k in self.increment_model.PARAM_NAMES}
        log_inc = self.increment_model.mixture_logpdf(dx, v, y, inc_theta)
        log_vol = self._volume_logpdf(v, y, theta)
        return log_inc + log_vol
