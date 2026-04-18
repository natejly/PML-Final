"""Gaussian latent-type model from Example 3.1 of the paper.

K = 3 trader types: informed, noise, manipulator. The conditional density of
the log-odds increment Delta X_t given volume v_t and outcome y is the
volume-gated mixture (Eq. 3):

    f_y(dx | v, theta) = sum_k rho_k(v; omega, gamma) * f_{k,y}(dx | v, theta_k)

with type-specific location-scale densities (Eq. 2). The orientation
constraint mu_1, mu_3 >= 0 (Remark 3.2) is enforced by the priors module via
softplus reparameterization.

All routines accept either NumPy arrays or PyTorch tensors and dispatch on the
backend; this lets the SMC and VI modules share a single model implementation.
"""

from __future__ import annotations

import math
from typing import Mapping, Any

import numpy as np
from scipy.special import gammaln as _np_lgamma
from scipy.special import logsumexp as _np_logsumexp


# Names of all theta parameters for serialization.
PARAM_NAMES = (
    "omega",   # (..., K)
    "gamma",   # (..., K, 2)
    "mu1",     # (...,)
    "lam1",    # (...,)
    "sigma1",  # (...,)
    "kappa1",  # (...,)
    "sigma2",  # (...,)
    "mu3",     # (...,)
    "tau3",    # (...,)
    "sigma3",  # (...,)
    "nu",      # (...,)
)


# ---------------------------------------------------------------------------
# Backend dispatch (numpy / torch)
# ---------------------------------------------------------------------------

def _is_torch(x):
    try:
        import torch
        return isinstance(x, torch.Tensor)
    except ImportError:
        return False


def _backend(x):
    if _is_torch(x):
        import torch
        return torch
    return np


def _logsumexp(x, axis):
    if _is_torch(x):
        import torch
        return torch.logsumexp(x, dim=axis)
    return _np_logsumexp(x, axis=axis)


def _lgamma(x):
    if _is_torch(x):
        import torch
        return torch.lgamma(x)
    return _np_lgamma(x)


def _log(x):
    return _backend(x).log(x)


def _exp(x):
    return _backend(x).exp(x)


def _sqrt(x):
    return _backend(x).sqrt(x)


def _where(cond, a, b):
    xp = _backend(cond) if hasattr(cond, "__array__") or _is_torch(cond) else _backend(a)
    return xp.where(cond, a, b)


# ---------------------------------------------------------------------------
# Type-specific log densities
# ---------------------------------------------------------------------------

_LOG_2PI = math.log(2.0 * math.pi)
_SCALE_FLOOR = 1e-3   # numerical floor on type scales to prevent overflow on
                       # extreme particles where softplus(z) is near zero


def _floor_scale(scale):
    """Apply a small numerical floor to a scale parameter."""
    if _is_torch(scale):
        return scale.clamp(min=_SCALE_FLOOR)
    return np.maximum(scale, _SCALE_FLOOR)


def _normal_logpdf(x, mean, scale):
    """Log density of N(mean, scale^2). Broadcasts over inputs."""
    scale = _floor_scale(scale)
    z = (x - mean) / scale
    return -0.5 * (z * z + _LOG_2PI) - _log(scale)


def _student_t_logpdf(x, mean, scale, nu):
    """Log density of location-scale Student-t with df nu > 0.

    Uses a stable form for the (nu+1)/2 * log(1 + z^2/nu) term to avoid
    overflow when z^2/nu is enormous (e.g. heavy tails, tiny scale).
    """
    scale = _floor_scale(scale)
    z = (x - mean) / scale
    half_nu = 0.5 * nu
    log_norm = (
        _lgamma(half_nu + 0.5)
        - _lgamma(half_nu)
        - 0.5 * _log(nu * math.pi)
    )
    # log(1 + z^2/nu) = log(nu + z^2) - log(nu); the right form avoids the
    # divide-by-tiny-nu and overflow when z is huge.
    z2 = z * z
    log1p_zsq_over_nu = _log(nu + z2) - _log(nu)
    return log_norm - 0.5 * (nu + 1.0) * log1p_zsq_over_nu - _log(scale)


def informed_logpdf(dx, v, y, mu1, lam1, sigma1, kappa1):
    """Type 1 (informed): N(mu1*(2y-1)*(1-exp(-lam1 v)), (sigma1/sqrt(1+kappa1 v))^2).

    mu1 etc. may carry a leading batch dimension; dx, v have shape (T,). The
    return shape is (..., T).
    """
    xp = _backend(dx) if not _is_torch(mu1) else _backend(mu1)
    sign = 2.0 * y - 1.0
    # Broadcast: place (T,) on the trailing axis; mu1 etc. take leading axes.
    if hasattr(mu1, "ndim") and mu1.ndim > 0:
        v_b = v[(None,) * mu1.ndim + (slice(None),)]
        dx_b = dx[(None,) * mu1.ndim + (slice(None),)]
        mu1 = mu1[..., None]
        lam1 = lam1[..., None]
        sigma1 = sigma1[..., None]
        kappa1 = kappa1[..., None]
    else:
        v_b = v
        dx_b = dx
    mean = mu1 * sign * (1.0 - _exp(-lam1 * v_b))
    scale = sigma1 / _sqrt(1.0 + kappa1 * v_b)
    return _normal_logpdf(dx_b, mean, scale)


def noise_logpdf(dx, v, sigma2):
    """Type 2 (noise): N(0, sigma2^2). Outcome-independent."""
    if hasattr(sigma2, "ndim") and sigma2.ndim > 0:
        dx_b = dx[(None,) * sigma2.ndim + (slice(None),)]
        sigma2 = sigma2[..., None]
    else:
        dx_b = dx
    zero = 0.0 * dx_b  # broadcast-safe zero
    return _normal_logpdf(dx_b, zero, sigma2)


def manipulator_logpdf(dx, v, y, mu3, tau3, sigma3, nu):
    """Type 3 (manipulator): location-scale Student-t.

    Mean = -mu3*(2y-1)*sigmoid(steep*(v - tau3)). The paper specifies a hard
    indicator 1{v>tau3}; we use a steep sigmoid so gradients through tau3 are
    well-defined for VI. With steep=20 and typical volume scales O(1) this is
    indistinguishable from the indicator outside a narrow boundary.
    """
    sign = 2.0 * y - 1.0
    if hasattr(mu3, "ndim") and mu3.ndim > 0:
        v_b = v[(None,) * mu3.ndim + (slice(None),)]
        dx_b = dx[(None,) * mu3.ndim + (slice(None),)]
        mu3 = mu3[..., None]
        tau3 = tau3[..., None]
        sigma3 = sigma3[..., None]
        nu = nu[..., None]
    else:
        v_b = v
        dx_b = dx
    steep = 20.0
    # sigmoid via stable formulation
    z = steep * (v_b - tau3)
    if _is_torch(z):
        import torch
        active = torch.sigmoid(z)
    else:
        active = 1.0 / (1.0 + np.exp(-z))
    mean = -mu3 * sign * active
    return _student_t_logpdf(dx_b, mean, sigma3, nu)


# ---------------------------------------------------------------------------
# Volume-dependent gating (softmax with log-volume logits)
# ---------------------------------------------------------------------------

def softmax_gate(v, omega, gamma):
    """Softmax gate with log-volume logits a_k(v;gamma) = gamma_{k,0} + gamma_{k,1} log(1+v).

    Parameters
    ----------
    v : (T,) array of volumes.
    omega : (..., K) base mixture weights (used as log-prior bias log omega_k).
    gamma : (..., K, 2) gating coefficients.

    Returns
    -------
    log_rho : (..., T, K) log mixture weights.
    """
    xp = _backend(omega)
    # Logits: log(omega_k) + gamma_{k,0} + gamma_{k,1} log(1+v)
    log_omega = _log(omega)  # (..., K)
    g0 = gamma[..., 0]       # (..., K)
    g1 = gamma[..., 1]       # (..., K)
    if hasattr(omega, "ndim") and omega.ndim >= 1:
        # Insert time axis between batch and K.
        # log_omega: (..., K) -> (..., 1, K)
        log_omega_t = log_omega[..., None, :]
        g0_t = g0[..., None, :]
        g1_t = g1[..., None, :]
        # v: (T,) -> (..., T, 1), respecting any leading batch dims
        if log_omega.ndim > 1:
            lead = log_omega.ndim - 1  # number of leading batch dims
            v_t = v[(None,) * lead + (slice(None), None)]
        else:
            v_t = v[:, None]
    else:
        log_omega_t = log_omega
        g0_t = g0
        g1_t = g1
        v_t = v[:, None]
    log1pv = _log(1.0 + v_t)
    logits = log_omega_t + g0_t + g1_t * log1pv
    log_rho = logits - _logsumexp(logits, axis=-1)[..., None]
    return log_rho


# ---------------------------------------------------------------------------
# Mixture log-pdf and full log-likelihood
# ---------------------------------------------------------------------------

def mixture_logpdf(dx, v, y, theta: Mapping[str, Any]):
    """log f_y(dx_t | v_t, theta) for each t. Returns shape (..., T)."""
    log_rho = softmax_gate(v, theta["omega"], theta["gamma"])  # (..., T, K)
    log_f1 = informed_logpdf(dx, v, y,
                             theta["mu1"], theta["lam1"], theta["sigma1"], theta["kappa1"])
    log_f2 = noise_logpdf(dx, v, theta["sigma2"])
    log_f3 = manipulator_logpdf(dx, v, y,
                                theta["mu3"], theta["tau3"], theta["sigma3"], theta["nu"])
    # Stack along K axis: each is (..., T) -> (..., T, K)
    if _is_torch(log_f1):
        import torch
        log_components = torch.stack([log_f1, log_f2, log_f3], dim=-1)
    else:
        log_components = np.stack([log_f1, log_f2, log_f3], axis=-1)
    return _logsumexp(log_rho + log_components, axis=-1)


def loglik(dx, v, y, theta: Mapping[str, Any]):
    """Full log-likelihood sum_t log f_y(dx_t | v_t, theta). Shape (...,)."""
    per_step = mixture_logpdf(dx, v, y, theta)  # (..., T)
    return per_step.sum(axis=-1)


# ---------------------------------------------------------------------------
# Convenience: stack a dict of per-particle scalars into batched theta
# ---------------------------------------------------------------------------

def stack_thetas(theta_list):
    """Stack a list of theta dicts (each scalar) into a single batched dict."""
    out = {}
    for name in PARAM_NAMES:
        arrs = [np.asarray(t[name]) for t in theta_list]
        out[name] = np.stack(arrs, axis=0)
    return out


def index_theta(theta, idx):
    """Extract a single theta dict at batch index `idx`."""
    return {name: np.asarray(theta[name])[idx] for name in PARAM_NAMES}
