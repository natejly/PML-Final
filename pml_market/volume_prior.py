"""Priors + bijectors for the volume-aware joint model.

Two flavors are provided:

  * ``VolumeLognormalPrior``: symmetric across Y in {0, 1}.

      mu_v[y]    ~ N(0, MU_V_PRIOR_SD^2)
      sigma_v[y] ~ HalfNormal(SIGMA_V_PRIOR_SD)

    sigma_v is parameterized as ``softplus(z)`` on the unconstrained scale.

  * ``VolumeLognormalEBPrior``: empirical-Bayes prior whose per-outcome
    hyperparameters are typically fit from a labeled panel
    (``from_panel(...)``).  This is what breaks the Y-symmetry that otherwise
    makes the volume term marginally Y-invariant.

      mu_v[y]            ~ N(mu_v_mean[y], mu_v_sd[y]^2)
      log sigma_v[y]     ~ N(log_sigma_mean[y], log_sigma_sd[y]^2)

    sigma_v is parameterized as ``exp(z)`` on the unconstrained scale, which
    pairs naturally with the lognormal anchor (mode and mean shift together).

Both priors lay out the unconstrained vector as
``z = [base_z, mu_v_z (2), sigma_v_z (2)]`` and add 4 dimensions to whatever
base prior you start with.
"""

from __future__ import annotations

import math
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .core import Prior
from .priors import (
    LatentTypePrior,
    _softplus, _softplus_inv, _log_softplus_jac,
)
from .model import _is_torch


def _as_like(arr, ref):
    """Cast a (2,) numpy array to torch tensor matching `ref` if needed."""
    if _is_torch(ref):
        import torch
        return torch.as_tensor(np.asarray(arr, dtype=np.float64),
                               dtype=ref.dtype, device=ref.device)
    return np.asarray(arr, dtype=np.float64)


# Default hyperparameters for the volume prior. Wide enough to cover typical
# Polymarket bucket volumes (log1p(v) ranging up to ~15 for v ~ 3M USD).
MU_V_PRIOR_SD = 10.0
SIGMA_V_PRIOR_SD = 5.0


class VolumeLognormalPrior(Prior):
    """Prior on (base params, mu_v, sigma_v) for the joint volume model.

    Parameters
    ----------
    base_prior : Prior, optional
        Prior on the increment-model parameters.  Defaults to
        `LatentTypePrior()`.
    mu_v_sd : float
        Std of the N(0, .) prior on mu_v[y].
    sigma_v_sd : float
        Scale of the HalfNormal(.) prior on sigma_v[y].
    """

    def __init__(self,
                 base_prior: Optional[Prior] = None,
                 mu_v_sd: float = MU_V_PRIOR_SD,
                 sigma_v_sd: float = SIGMA_V_PRIOR_SD):
        self.base = base_prior or LatentTypePrior()
        self.mu_v_sd = float(mu_v_sd)
        self.sigma_v_sd = float(sigma_v_sd)
        self.UNCONSTRAINED_DIM = self.base.UNCONSTRAINED_DIM + 4

    def __repr__(self) -> str:
        return (f"VolumeLognormalPrior(base={type(self.base).__name__}, "
                f"mu_v_sd={self.mu_v_sd}, sigma_v_sd={self.sigma_v_sd})")

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, rng: np.random.Generator, n: int = 1) -> Mapping[str, np.ndarray]:
        base_theta = self.base.sample(rng, n)
        if n == 1:
            mu_v = rng.normal(0.0, self.mu_v_sd, size=2)
            sigma_v = np.abs(rng.normal(0.0, self.sigma_v_sd, size=2))
        else:
            mu_v = rng.normal(0.0, self.mu_v_sd, size=(n, 2))
            sigma_v = np.abs(rng.normal(0.0, self.sigma_v_sd, size=(n, 2)))
        return {**base_theta, "mu_v": mu_v, "sigma_v": sigma_v}

    # ------------------------------------------------------------------
    # Bijector R^d -> theta
    # ------------------------------------------------------------------

    def transform(self, z):
        base_dim = self.base.UNCONSTRAINED_DIM
        base_z = z[..., :base_dim]
        vol_z = z[..., base_dim:]                 # (..., 4)

        base_theta, base_jac = self.base.transform(base_z)

        mu_v = vol_z[..., 0:2]                    # identity
        sig_z = vol_z[..., 2:4]
        sigma_v = _softplus(sig_z)

        # Sum log|softplus'(z)| over the two sigma_v dims.
        vol_jac = _log_softplus_jac(sig_z).sum(axis=-1)

        theta = {**base_theta, "mu_v": mu_v, "sigma_v": sigma_v}
        return theta, base_jac + vol_jac

    def to_unconstrained(self, theta: Mapping[str, Any]) -> np.ndarray:
        base_z = self.base.to_unconstrained(theta)
        mu_v = np.asarray(theta["mu_v"])          # (..., 2)
        sigma_v = np.asarray(theta["sigma_v"])    # (..., 2)
        sig_z = _softplus_inv(sigma_v)
        vol_z = np.concatenate([mu_v, sig_z], axis=-1)
        return np.concatenate([base_z, vol_z], axis=-1)

    # ------------------------------------------------------------------
    # log pi(z) for VI / SMC rejuvenation
    # ------------------------------------------------------------------

    def log_prior_unconstrained(self, z):
        base_dim = self.base.UNCONSTRAINED_DIM
        base_z = z[..., :base_dim]
        vol_z = z[..., base_dim:]

        base_lp = self.base.log_prior_unconstrained(base_z)

        mu_v = vol_z[..., 0:2]
        sig_z = vol_z[..., 2:4]
        sigma_v = _softplus(sig_z)
        vol_jac = _log_softplus_jac(sig_z).sum(axis=-1)

        # log Pi(mu_v): two iid N(0, mu_v_sd^2)
        mu_const = 0.5 * math.log(2.0 * math.pi) + math.log(self.mu_v_sd)
        if _is_torch(mu_v):
            log_pi_mu = (-0.5 * (mu_v / self.mu_v_sd) ** 2).sum(dim=-1) - 2.0 * mu_const
        else:
            log_pi_mu = (-0.5 * (mu_v / self.mu_v_sd) ** 2).sum(axis=-1) - 2.0 * mu_const

        # log Pi(sigma_v): two iid HalfNormal(sigma_v_sd) on the constrained scale
        half_const = 0.5 * math.log(2.0 / math.pi) - math.log(self.sigma_v_sd)
        if _is_torch(sigma_v):
            log_pi_sig = (-0.5 * (sigma_v / self.sigma_v_sd) ** 2).sum(dim=-1) + 2.0 * half_const
        else:
            log_pi_sig = (-0.5 * (sigma_v / self.sigma_v_sd) ** 2).sum(axis=-1) + 2.0 * half_const

        return base_lp + log_pi_mu + log_pi_sig + vol_jac


# ---------------------------------------------------------------------------
# Empirical-Bayes / asymmetric prior
# ---------------------------------------------------------------------------

class VolumeLognormalEBPrior(Prior):
    """Asymmetric (per-Y) empirical-Bayes prior for the volume model.

    Hyperparameters are length-2 arrays indexed by Y in {0, 1}:

        mu_v[y]            ~ N(mu_v_mean[y], mu_v_sd[y]^2)
        log sigma_v[y]     ~ N(log_sigma_mean[y], log_sigma_sd[y]^2)

    The unconstrained reparameterization is

        mu_v[y]    = z_mu[y]                      (identity)
        sigma_v[y] = exp(z_sigma[y])              (lognormal)

    so the Jacobian contribution from the volume block is sum_y z_sigma[y].

    Use :meth:`from_panel` to fit the four (2,) arrays from a labeled panel
    of trajectories (raw Yes/No outcomes, no winner alignment).
    """

    def __init__(
        self,
        base_prior: Optional[Prior] = None,
        mu_v_mean: Sequence[float] = (0.0, 0.0),
        mu_v_sd: Sequence[float] = (10.0, 10.0),
        log_sigma_mean: Sequence[float] = (0.0, 0.0),
        log_sigma_sd: Sequence[float] = (1.0, 1.0),
    ):
        self.base = base_prior or LatentTypePrior()
        self.mu_v_mean = np.asarray(mu_v_mean, dtype=np.float64).reshape(2)
        self.mu_v_sd = np.asarray(mu_v_sd, dtype=np.float64).reshape(2)
        self.log_sigma_mean = np.asarray(log_sigma_mean, dtype=np.float64).reshape(2)
        self.log_sigma_sd = np.asarray(log_sigma_sd, dtype=np.float64).reshape(2)
        if np.any(self.mu_v_sd <= 0) or np.any(self.log_sigma_sd <= 0):
            raise ValueError("mu_v_sd and log_sigma_sd must be strictly positive")
        self.UNCONSTRAINED_DIM = self.base.UNCONSTRAINED_DIM + 4

    def __repr__(self) -> str:
        return (
            f"VolumeLognormalEBPrior(base={type(self.base).__name__}, "
            f"mu_v_mean={self.mu_v_mean.tolist()}, mu_v_sd={self.mu_v_sd.tolist()}, "
            f"log_sigma_mean={self.log_sigma_mean.tolist()}, "
            f"log_sigma_sd={self.log_sigma_sd.tolist()})"
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, rng: np.random.Generator, n: int = 1) -> Mapping[str, np.ndarray]:
        base_theta = self.base.sample(rng, n)
        if n == 1:
            mu_v = rng.normal(self.mu_v_mean, self.mu_v_sd, size=2)
            log_sig = rng.normal(self.log_sigma_mean, self.log_sigma_sd, size=2)
            sigma_v = np.exp(log_sig)
        else:
            mu_v = (self.mu_v_mean
                    + self.mu_v_sd * rng.standard_normal(size=(n, 2)))
            log_sig = (self.log_sigma_mean
                       + self.log_sigma_sd * rng.standard_normal(size=(n, 2)))
            sigma_v = np.exp(log_sig)
        return {**base_theta, "mu_v": mu_v, "sigma_v": sigma_v}

    # ------------------------------------------------------------------
    # Bijector R^d -> theta
    # ------------------------------------------------------------------

    def transform(self, z):
        base_dim = self.base.UNCONSTRAINED_DIM
        base_z = z[..., :base_dim]
        vol_z = z[..., base_dim:]                 # (..., 4)

        base_theta, base_jac = self.base.transform(base_z)

        mu_v = vol_z[..., 0:2]                    # identity
        sig_z = vol_z[..., 2:4]                   # log sigma_v
        if _is_torch(sig_z):
            import torch
            sigma_v = torch.exp(sig_z)
            vol_jac = sig_z.sum(dim=-1)
        else:
            sigma_v = np.exp(sig_z)
            vol_jac = sig_z.sum(axis=-1)

        theta = {**base_theta, "mu_v": mu_v, "sigma_v": sigma_v}
        return theta, base_jac + vol_jac

    def to_unconstrained(self, theta: Mapping[str, Any]) -> np.ndarray:
        base_z = self.base.to_unconstrained(theta)
        mu_v = np.asarray(theta["mu_v"])          # (..., 2)
        sigma_v = np.asarray(theta["sigma_v"])    # (..., 2)
        sig_z = np.log(np.clip(sigma_v, 1e-12, None))
        vol_z = np.concatenate([mu_v, sig_z], axis=-1)
        return np.concatenate([base_z, vol_z], axis=-1)

    # ------------------------------------------------------------------
    # log pi(z) for VI / SMC rejuvenation
    # ------------------------------------------------------------------

    def log_prior_unconstrained(self, z):
        base_dim = self.base.UNCONSTRAINED_DIM
        base_z = z[..., :base_dim]
        vol_z = z[..., base_dim:]

        base_lp = self.base.log_prior_unconstrained(base_z)

        mu_v = vol_z[..., 0:2]
        sig_z = vol_z[..., 2:4]                  # log sigma_v

        mu_mean = _as_like(self.mu_v_mean, mu_v)
        mu_sd   = _as_like(self.mu_v_sd, mu_v)
        ls_mean = _as_like(self.log_sigma_mean, sig_z)
        ls_sd   = _as_like(self.log_sigma_sd, sig_z)

        # log Pi(mu_v): N(mu_mean[y], mu_sd[y]^2), summed over y.
        if _is_torch(mu_v):
            import torch
            log_pi_mu = (
                -0.5 * ((mu_v - mu_mean) / mu_sd) ** 2
                - 0.5 * math.log(2.0 * math.pi)
                - torch.log(mu_sd)
            ).sum(dim=-1)
            log_pi_sig = (
                -0.5 * ((sig_z - ls_mean) / ls_sd) ** 2
                - 0.5 * math.log(2.0 * math.pi)
                - torch.log(ls_sd)
            ).sum(dim=-1)
        else:
            log_pi_mu = (
                -0.5 * ((mu_v - mu_mean) / mu_sd) ** 2
                - 0.5 * math.log(2.0 * math.pi)
                - np.log(mu_sd)
            ).sum(axis=-1)
            log_pi_sig = (
                -0.5 * ((sig_z - ls_mean) / ls_sd) ** 2
                - 0.5 * math.log(2.0 * math.pi)
                - np.log(ls_sd)
            ).sum(axis=-1)

        # The lognormal Jacobian for sigma_v = exp(sig_z) is exp(sig_z) = sigma_v;
        # log|det| = sig_z, summed over the two volume dims.
        if _is_torch(sig_z):
            vol_jac = sig_z.sum(dim=-1)
        else:
            vol_jac = sig_z.sum(axis=-1)

        return base_lp + log_pi_mu + log_pi_sig + vol_jac

    # ------------------------------------------------------------------
    # Empirical-Bayes fitter
    # ------------------------------------------------------------------

    @classmethod
    def from_panel(
        cls,
        trajectories: Sequence[Mapping[str, Any]],
        base_prior: Optional[Prior] = None,
        mu_sd_floor: float = 0.25,
        log_sigma_sd_floor: float = 0.25,
        shrinkage: float = 1.0,
    ) -> Tuple["VolumeLognormalEBPrior", Mapping[str, Any]]:
        """Fit the four (2,) hyperparameter arrays from labeled trajectories.

        The trajectories must be the dicts produced by
        ``data.fetch_resolved_binary_markets`` (or ``build_trajectory``):
        each must contain a ``"volumes"`` array and a ``"winner_label"``
        whose value is ``"Yes"`` or ``"No"``.  The volume data itself is
        invariant to winner alignment, so we use the *raw* Yes/No label to
        split the panel into two groups.

        Method-of-moments per group ``Y in {0=No, 1=Yes}``:

            For each market m in group Y, compute
                m_m = mean_t   log(1 + v_t)
                s_m = std_t    log(1 + v_t)

            mu_v_mean[Y]    = mean_m m_m
            mu_v_sd[Y]      = max( shrinkage * std_m m_m,  mu_sd_floor )
            log_sigma_mean[Y] = mean_m log(s_m)
            log_sigma_sd[Y] = max( shrinkage * std_m log(s_m), log_sigma_sd_floor )

        Returns the fitted prior plus a small diagnostic dict (per-group
        sample counts and raw moments) so the caller can sanity check.
        """
        groups: List[List[np.ndarray]] = [[], []]
        skipped = 0
        for traj in trajectories:
            wl = traj.get("winner_label")
            if wl not in ("Yes", "No"):
                skipped += 1
                continue
            y_raw = 1 if wl == "Yes" else 0
            v = np.asarray(traj["volumes"], dtype=np.float64)
            if v.size < 2:
                skipped += 1
                continue
            groups[y_raw].append(v)

        if not groups[0] or not groups[1]:
            raise ValueError(
                "from_panel requires at least one Yes-resolving and one "
                "No-resolving market; got "
                f"|No|={len(groups[0])}, |Yes|={len(groups[1])}"
            )

        mu_means = np.zeros(2)
        mu_sds = np.zeros(2)
        ls_means = np.zeros(2)
        ls_sds = np.zeros(2)
        diag = {"counts": [len(groups[0]), len(groups[1])],
                "per_market_means": [[], []],
                "per_market_log_stds": [[], []]}
        for y in (0, 1):
            per_market_m = []
            per_market_logs = []
            for v in groups[y]:
                lv = np.log1p(v)
                per_market_m.append(float(lv.mean()))
                # ddof=0 to keep things finite for short markets.
                s = float(lv.std(ddof=0))
                per_market_logs.append(math.log(max(s, 1e-3)))
            arr_m = np.asarray(per_market_m)
            arr_s = np.asarray(per_market_logs)
            mu_means[y] = arr_m.mean()
            ls_means[y] = arr_s.mean()
            sd_m = arr_m.std(ddof=0) if arr_m.size > 1 else 0.0
            sd_s = arr_s.std(ddof=0) if arr_s.size > 1 else 0.0
            mu_sds[y] = max(shrinkage * sd_m, mu_sd_floor)
            ls_sds[y] = max(shrinkage * sd_s, log_sigma_sd_floor)
            diag["per_market_means"][y] = per_market_m
            diag["per_market_log_stds"][y] = per_market_logs

        diag["mu_v_mean"] = mu_means.tolist()
        diag["mu_v_sd"] = mu_sds.tolist()
        diag["log_sigma_mean"] = ls_means.tolist()
        diag["log_sigma_sd"] = ls_sds.tolist()
        diag["skipped"] = skipped

        prior = cls(
            base_prior=base_prior,
            mu_v_mean=mu_means,
            mu_v_sd=mu_sds,
            log_sigma_mean=ls_means,
            log_sigma_sd=ls_sds,
        )
        return prior, diag
