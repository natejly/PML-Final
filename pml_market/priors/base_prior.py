"""Priors and unconstrained reparameterization for the Gaussian latent-type model.

Prior factorization (Section 3.8 + Remark 3.2 + Proposition A.1 in the paper):

  omega       ~ Dirichlet(alpha = (1, 1, 1))         on the 2-simplex
  gamma_{k,j} ~ N(0, 1)                              for k=1..K=3, j=0,1
  mu1, mu3    ~ HalfNormal(0.5)                      enforcing orientation
  lam1        ~ HalfNormal(1)
  kappa1      ~ HalfNormal(1)
  sigma_k     ~ HalfNormal(1)                        for k=1,2,3
  tau3        ~ HalfNormal(5)
  nu - 2      ~ Gamma(shape=2, scale=2)              so nu > 2 (Asn. 4.5)

For VI we map theta -> an unconstrained vector z in R^17 via:
  - omega:    additive log-ratio (ALR) with reference component omega_K (K-1=2 reals)
  - gamma:    identity (6 reals)
  - positives (mu1, lam1, sigma1, kappa1, sigma2, mu3, tau3, sigma3, nu_shift):
              softplus^{-1} (9 reals).  nu = 2 + softplus(z).

The bijector returns log|det d theta / d z| so that the change of variables
  log pi(z) = log Pi(theta(z)) + log|det d theta / d z|
holds, which is what VI consumes.
"""

from __future__ import annotations

import math
from typing import Dict, Any

import numpy as np
from scipy.special import gammaln as _np_lgamma

from ..models.base_model import _is_torch, _backend, PARAM_NAMES


# ---------------------------------------------------------------------------
# Prior hyperparameters
# ---------------------------------------------------------------------------

DIRICHLET_ALPHA = np.array([1.0, 1.0, 1.0])
GAMMA_PRIOR_SD = 1.0          # gamma_{k,j} ~ N(0, GAMMA_PRIOR_SD^2)
MU_PRIOR_SD = 0.5             # mu1, mu3 ~ HalfNormal(MU_PRIOR_SD)
LAM_PRIOR_SD = 1.0            # lam1 ~ HalfNormal
KAPPA_PRIOR_SD = 1.0
SIGMA_PRIOR_SD = 1.0
TAU_PRIOR_SD = 5.0
NU_GAMMA_SHAPE = 2.0          # nu - 2 ~ Gamma(NU_GAMMA_SHAPE, scale=NU_GAMMA_SCALE)
NU_GAMMA_SCALE = 2.0
NU_FLOOR = 2.0

# Unconstrained dimension layout (in this order)
UNCONSTRAINED_LAYOUT = (
    ("omega_alr", 2),  # K-1 reals
    ("gamma", 6),      # 3 types x 2 coefficients, flattened row-major
    ("mu1", 1),
    ("lam1", 1),
    ("sigma1", 1),
    ("kappa1", 1),
    ("sigma2", 1),
    ("mu3", 1),
    ("tau3", 1),
    ("sigma3", 1),
    ("nu_shift", 1),
)
UNCONSTRAINED_DIM = sum(d for _, d in UNCONSTRAINED_LAYOUT)


# ---------------------------------------------------------------------------
# Helpers (numpy)
# ---------------------------------------------------------------------------

def _halfnormal_logpdf(x, sd):
    """log pdf of HalfNormal(sd) on x >= 0."""
    if np.any(np.asarray(x) < 0):
        return -np.inf
    return 0.5 * math.log(2.0 / math.pi) - math.log(sd) - 0.5 * (x / sd) ** 2


def _gamma_shifted_logpdf(nu, shape, scale, shift):
    """log pdf of (nu - shift) ~ Gamma(shape, scale)."""
    z = nu - shift
    if np.any(np.asarray(z) <= 0):
        return -np.inf
    return (
        (shape - 1.0) * np.log(z)
        - z / scale
        - shape * math.log(scale)
        - _np_lgamma(shape)
    )


def _dirichlet_logpdf(omega, alpha):
    """log pdf of Dirichlet(alpha) on a probability vector omega."""
    omega = np.asarray(omega)
    alpha = np.asarray(alpha)
    if np.any(omega <= 0) or not np.isclose(omega.sum(), 1.0, atol=1e-6):
        return -np.inf
    log_B = _np_lgamma(alpha).sum() - _np_lgamma(alpha.sum())
    return ((alpha - 1.0) * np.log(omega)).sum() - log_B


def _normal_logpdf_np(x, mean, sd):
    return -0.5 * ((x - mean) / sd) ** 2 - 0.5 * math.log(2.0 * math.pi) - math.log(sd)


# ---------------------------------------------------------------------------
# Prior log-density on the constrained theta dict (numpy only)
# ---------------------------------------------------------------------------

def log_prior(theta: Dict[str, Any]) -> float:
    """Log Pi(theta), a real number; returns -inf if theta violates support."""
    lp = 0.0
    lp += _dirichlet_logpdf(theta["omega"], DIRICHLET_ALPHA)
    lp += _normal_logpdf_np(np.asarray(theta["gamma"]), 0.0, GAMMA_PRIOR_SD).sum()
    lp += _halfnormal_logpdf(theta["mu1"], MU_PRIOR_SD)
    lp += _halfnormal_logpdf(theta["lam1"], LAM_PRIOR_SD)
    lp += _halfnormal_logpdf(theta["sigma1"], SIGMA_PRIOR_SD)
    lp += _halfnormal_logpdf(theta["kappa1"], KAPPA_PRIOR_SD)
    lp += _halfnormal_logpdf(theta["sigma2"], SIGMA_PRIOR_SD)
    lp += _halfnormal_logpdf(theta["mu3"], MU_PRIOR_SD)
    lp += _halfnormal_logpdf(theta["tau3"], TAU_PRIOR_SD)
    lp += _halfnormal_logpdf(theta["sigma3"], SIGMA_PRIOR_SD)
    lp += _gamma_shifted_logpdf(theta["nu"], NU_GAMMA_SHAPE, NU_GAMMA_SCALE, NU_FLOOR)
    return float(lp)


def log_prior_batched(theta: Dict[str, np.ndarray]) -> np.ndarray:
    """Vectorized log Pi for a leading batch dim. Each theta value has shape
    (N,...). Returns shape (N,)."""
    N = np.asarray(theta["mu1"]).shape[0]
    out = np.zeros(N, dtype=np.float64)
    for i in range(N):
        ti = {k: np.asarray(theta[k])[i] for k in PARAM_NAMES}
        out[i] = log_prior(ti)
    return out


# ---------------------------------------------------------------------------
# Sampling from the prior
# ---------------------------------------------------------------------------

def sample_prior(rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Draw a single theta from Pi."""
    K = 3
    return dict(
        omega=rng.dirichlet(DIRICHLET_ALPHA),
        gamma=rng.normal(0.0, GAMMA_PRIOR_SD, size=(K, 2)),
        mu1=np.array(np.abs(rng.normal(0.0, MU_PRIOR_SD))),
        lam1=np.array(np.abs(rng.normal(0.0, LAM_PRIOR_SD))),
        sigma1=np.array(np.abs(rng.normal(0.0, SIGMA_PRIOR_SD))),
        kappa1=np.array(np.abs(rng.normal(0.0, KAPPA_PRIOR_SD))),
        sigma2=np.array(np.abs(rng.normal(0.0, SIGMA_PRIOR_SD))),
        mu3=np.array(np.abs(rng.normal(0.0, MU_PRIOR_SD))),
        tau3=np.array(np.abs(rng.normal(0.0, TAU_PRIOR_SD))),
        sigma3=np.array(np.abs(rng.normal(0.0, SIGMA_PRIOR_SD))),
        nu=np.array(NU_FLOOR + rng.gamma(NU_GAMMA_SHAPE, NU_GAMMA_SCALE)),
    )


def sample_prior_batched(rng: np.random.Generator, n: int) -> Dict[str, np.ndarray]:
    """Draw n iid thetas from Pi as a batched dict."""
    samples = [sample_prior(rng) for _ in range(n)]
    out = {}
    for name in PARAM_NAMES:
        out[name] = np.stack([np.asarray(s[name]) for s in samples], axis=0)
    return out


# ---------------------------------------------------------------------------
# Unconstrained reparameterization (numpy / torch dispatch)
# ---------------------------------------------------------------------------

def _softplus(z):
    if _is_torch(z):
        import torch
        return torch.nn.functional.softplus(z)
    return np.logaddexp(0.0, z)


def _softplus_inv(x):
    """Inverse softplus: solve softplus(z) = x, i.e. z = log(exp(x) - 1)."""
    x = np.asarray(x, dtype=np.float64)
    return np.log(np.expm1(np.clip(x, 1e-10, None)))


def _log_softplus_jac(z):
    """log|d softplus(z) / dz| = log sigmoid(z) = -softplus(-z)."""
    return -_softplus(-z)


def _alr_to_simplex(z):
    """ALR (K-1 reals z) -> probability vector omega in R^K. K is inferred."""
    if _is_torch(z):
        import torch
        zeros = torch.zeros(z.shape[:-1] + (1,), dtype=z.dtype, device=z.device)
        full = torch.cat([z, zeros], dim=-1)
        return torch.softmax(full, dim=-1)
    zeros = np.zeros(z.shape[:-1] + (1,))
    full = np.concatenate([z, zeros], axis=-1)
    full = full - full.max(axis=-1, keepdims=True)
    e = np.exp(full)
    return e / e.sum(axis=-1, keepdims=True)


def _alr_log_jac(z):
    """log|det dω/dz| for the ALR-with-reference parameterization.

    For omega = softmax([z, 0]), the K-1 x K-1 Jacobian determinant equals
    prod_k omega_k (a standard result). So log|det J| = sum_k log omega_k.
    """
    omega = _alr_to_simplex(z)
    return _backend(omega).log(omega).sum(axis=-1)


# ---------------------------------------------------------------------------
# Forward bijector: z (R^17) -> (theta dict, log|det dθ/dz|)
# ---------------------------------------------------------------------------

def transform(z):
    """Map an unconstrained vector (or batch) to a theta dict.

    Parameters
    ----------
    z : array of shape (..., UNCONSTRAINED_DIM) (numpy or torch).

    Returns
    -------
    theta : dict of arrays with leading batch shape matching z.shape[:-1].
    log_det_jac : array of shape z.shape[:-1] giving log|det dθ/dz|.
    """
    xp = _backend(z)
    K = 3

    # Slice unconstrained components
    idx = 0
    pieces = {}
    for name, d in UNCONSTRAINED_LAYOUT:
        pieces[name] = z[..., idx : idx + d] if d > 1 else z[..., idx]
        idx += d
    assert idx == UNCONSTRAINED_DIM

    # omega via ALR
    omega = _alr_to_simplex(pieces["omega_alr"])  # (..., K)
    log_jac = _alr_log_jac(pieces["omega_alr"])   # (...)

    # gamma: identity reshape to (..., K, 2)
    gamma_flat = pieces["gamma"]  # (..., 6)
    if _is_torch(gamma_flat):
        gamma = gamma_flat.reshape(gamma_flat.shape[:-1] + (K, 2))
    else:
        gamma = gamma_flat.reshape(gamma_flat.shape[:-1] + (K, 2))
    # log|det J| = 0 for identity

    # positives via softplus
    pos_names = ("mu1", "lam1", "sigma1", "kappa1", "sigma2",
                 "mu3", "tau3", "sigma3")
    pos_vals = {}
    for nm in pos_names:
        zi = pieces[nm]
        pos_vals[nm] = _softplus(zi)
        log_jac = log_jac + _log_softplus_jac(zi)

    # nu = NU_FLOOR + softplus(z); Jacobian = softplus'(z)
    z_nu = pieces["nu_shift"]
    nu = NU_FLOOR + _softplus(z_nu)
    log_jac = log_jac + _log_softplus_jac(z_nu)

    theta = dict(
        omega=omega,
        gamma=gamma,
        mu1=pos_vals["mu1"], lam1=pos_vals["lam1"],
        sigma1=pos_vals["sigma1"], kappa1=pos_vals["kappa1"],
        sigma2=pos_vals["sigma2"],
        mu3=pos_vals["mu3"], tau3=pos_vals["tau3"], sigma3=pos_vals["sigma3"],
        nu=nu,
    )
    return theta, log_jac


def to_unconstrained(theta: Dict[str, Any]) -> np.ndarray:
    """Inverse of transform: theta dict -> unconstrained vector (numpy)."""
    omega = np.asarray(theta["omega"])
    K = omega.shape[-1]
    # ALR with omega_K as reference: z_k = log(omega_k / omega_K)
    z_omega = np.log(omega[..., :K - 1] / omega[..., K - 1:K])
    z_gamma = np.asarray(theta["gamma"]).reshape(omega.shape[:-1] + (K * 2,))
    pieces = [z_omega, z_gamma]
    for nm in ("mu1", "lam1", "sigma1", "kappa1", "sigma2",
               "mu3", "tau3", "sigma3"):
        pieces.append(_softplus_inv(np.asarray(theta[nm])).reshape(omega.shape[:-1] + (1,)))
    pieces.append(
        _softplus_inv(np.asarray(theta["nu"]) - NU_FLOOR).reshape(omega.shape[:-1] + (1,))
    )
    return np.concatenate(pieces, axis=-1)


# ---------------------------------------------------------------------------
# Log prior in unconstrained space (for VI)
# ---------------------------------------------------------------------------

def log_prior_unconstrained(z) -> Any:
    """log pi(z) = log Pi(theta(z)) + log|det dθ/dz|, working in numpy or torch.

    For VI this is the (constant) target prior density evaluated at samples z
    drawn from the variational q(z;phi). It is differentiable in z.
    """
    theta, log_jac = transform(z)
    xp = _backend(z)

    lp = 0.0 * log_jac  # broadcast-safe zero

    # Dirichlet prior on omega
    K = 3
    log_B = _np_lgamma(DIRICHLET_ALPHA).sum() - _np_lgamma(DIRICHLET_ALPHA.sum())
    if _is_torch(theta["omega"]):
        import torch
        alpha = torch.as_tensor(DIRICHLET_ALPHA, dtype=theta["omega"].dtype,
                                device=theta["omega"].device)
        lp = lp + ((alpha - 1.0) * theta["omega"].log()).sum(dim=-1) - log_B
    else:
        lp = lp + ((DIRICHLET_ALPHA - 1.0) * np.log(theta["omega"])).sum(axis=-1) - log_B

    # N(0,1) on gamma (all 6 components)
    if _is_torch(theta["gamma"]):
        gamma_sq = (theta["gamma"] ** 2).sum(dim=(-2, -1))
    else:
        gamma_sq = (theta["gamma"] ** 2).sum(axis=(-2, -1))
    lp = lp + (-0.5 * gamma_sq - 6.0 * (0.5 * math.log(2.0 * math.pi) + math.log(GAMMA_PRIOR_SD)))

    # HalfNormal priors on positives
    half_norm_const = lambda sd: 0.5 * math.log(2.0 / math.pi) - math.log(sd)
    for nm, sd in (("mu1", MU_PRIOR_SD), ("lam1", LAM_PRIOR_SD),
                   ("sigma1", SIGMA_PRIOR_SD), ("kappa1", KAPPA_PRIOR_SD),
                   ("sigma2", SIGMA_PRIOR_SD), ("mu3", MU_PRIOR_SD),
                   ("tau3", TAU_PRIOR_SD), ("sigma3", SIGMA_PRIOR_SD)):
        x = theta[nm]
        lp = lp + half_norm_const(sd) - 0.5 * (x / sd) ** 2

    # nu - 2 ~ Gamma(shape=2, scale=2)
    z_nu = theta["nu"] - NU_FLOOR
    if _is_torch(z_nu):
        log_z = z_nu.log()
    else:
        log_z = np.full_like(np.asarray(z_nu), -np.inf, dtype=np.float64)
        mask = np.asarray(z_nu) > 0
        log_z = np.where(mask, np.log(np.maximum(z_nu, np.finfo(float).tiny)), log_z)
    lp = lp + (
        (NU_GAMMA_SHAPE - 1.0) * log_z
        - z_nu / NU_GAMMA_SCALE
        - NU_GAMMA_SHAPE * math.log(NU_GAMMA_SCALE)
        - _np_lgamma(NU_GAMMA_SHAPE)
    )

    return lp + log_jac


# ---------------------------------------------------------------------------
# OOP classes
# ---------------------------------------------------------------------------

from ..core import Prior  # noqa: E402  (placed after free functions to avoid cycles)


class BasePrior(Prior):
    """Primary prior + bijector implementation for the base model."""

    UNCONSTRAINED_DIM = UNCONSTRAINED_DIM

    log_prior = staticmethod(log_prior)
    log_prior_batched = staticmethod(log_prior_batched)
    sample_prior = staticmethod(sample_prior)
    sample_prior_batched = staticmethod(sample_prior_batched)
    transform = staticmethod(transform)
    to_unconstrained = staticmethod(to_unconstrained)
    log_prior_unconstrained = staticmethod(log_prior_unconstrained)

    def sample(self, rng: np.random.Generator, n: int = 1):
        """Draw ``n`` iid samples, matching the ``Prior`` interface."""
        return sample_prior_batched(rng, int(n))
