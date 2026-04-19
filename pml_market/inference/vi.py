"""Mean-field reparameterized variational inference for the Bayes factor (Algorithm 2).

We approximate the posterior over the unconstrained parameter z in R^d
(determined by the prior's `UNCONSTRAINED_DIM`) with a diagonal Gaussian
q(z; phi) = N(mu_phi, diag(sigma_phi^2)).  The ELBO under outcome y is

    L(phi; y) = E_{z ~ q} [log p(Delta x | v, y, theta(z)) + log pi(z)] + H[q],

where pi(z) is the prior in unconstrained space (returned by
`prior.log_prior_unconstrained`) and H[q] is the differential entropy of q.
We optimize phi by stochastic gradient ascent (Adam) with the
reparameterization trick.

By Eq. 7 the (approximate) log Bayes factor is L^*(1) - L^*(0).  Per
Remark 5.1 this is biased (the difference of two ELBOs is not bounded by
log BF in either direction), so callers should treat the resulting posterior
as approximate.

Both the model and the prior are injected, so the VI engine works on any
(Model, Prior) subclass pair.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..core import Inference, Model, Prior


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _gaussian_entropy(log_sigma: torch.Tensor) -> torch.Tensor:
    """Entropy of N(mu, diag(sigma^2)).  Returns scalar.

    H = sum_d (0.5 * log(2*pi*e) + log sigma_d).
    """
    d = log_sigma.shape[-1]
    return 0.5 * d * (1.0 + math.log(2.0 * math.pi)) + log_sigma.sum()


def _elbo_estimate(mu: torch.Tensor, log_sigma: torch.Tensor,
                   dx_t: torch.Tensor, v_t: torch.Tensor, y: int,
                   n_samples: int,
                   model: Model, prior: Prior) -> torch.Tensor:
    """Single-batch reparameterized estimate of L(phi; y)."""
    sigma = log_sigma.exp()
    eps = torch.randn(n_samples, mu.shape[0], dtype=mu.dtype, device=mu.device)
    z = mu + sigma * eps                                     # (S, d)

    theta_batch, _ = prior.transform(z)
    log_lik = model.loglik(dx_t, v_t, y, theta_batch)        # (S,)
    log_prior_z = prior.log_prior_unconstrained(z)           # (S,)
    expected_joint = (log_lik + log_prior_z).mean()

    return expected_joint + _gaussian_entropy(log_sigma)


def _stable_sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# ---------------------------------------------------------------------------
# VIInference class (public API)
# ---------------------------------------------------------------------------


class VIInference(Inference):
    """Algorithm 2: mean-field VI estimator + Bayes factor wrapper.

    Hyperparameters live on the instance; algorithm-agnostic arguments such
    as the prior odds ``pi0`` and the random ``seed`` are passed at call time.

    Parameters
    ----------
    n_steps : int
        Number of Adam steps per outcome.
    n_samples : int
        Reparameterized samples per ELBO estimate.
    learning_rate : float
        Adam learning rate.
    init_log_sigma : float
        Initial value for the diagonal log-std (broadcast to all dims).
    final_n_samples : int
        Number of samples used for the final, less-noisy ELBO evaluation.
    grad_clip : float
        Max-norm clip on the gradient of (mu, log_sigma).
    progress : bool
        Show a tqdm progress bar.
    """

    def __init__(
        self,
        n_steps: int = 1500,
        n_samples: int = 8,
        learning_rate: float = 0.05,
        init_log_sigma: float = -1.5,
        final_n_samples: int = 128,
        grad_clip: float = 10.0,
        progress: bool = False,
    ):
        self.n_steps = int(n_steps)
        self.n_samples = int(n_samples)
        self.learning_rate = float(learning_rate)
        self.init_log_sigma = float(init_log_sigma)
        self.final_n_samples = int(final_n_samples)
        self.grad_clip = float(grad_clip)
        self.progress = bool(progress)

    def __repr__(self) -> str:
        return (f"VIInference(n_steps={self.n_steps}, n_samples={self.n_samples}, "
                f"lr={self.learning_rate})")

    # ------------------------------------------------------------------
    # Single-outcome ELBO maximization (Algorithm 2)
    # ------------------------------------------------------------------

    def marginal_likelihood(
        self,
        dx: np.ndarray,
        v: np.ndarray,
        y: int,
        model: Model,
        prior: Prior,
        *,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run mean-field VI to maximize L(phi; y).

        Returns
        -------
        dict
            ``elbo``, ``mu``, ``log_sigma``, ``theta_mean``, ``elbo_trace``.
        """
        if seed is not None:
            torch.manual_seed(seed)

        d = prior.UNCONSTRAINED_DIM
        dtype = torch.float64
        device = torch.device("cpu")

        mu = torch.zeros(d, dtype=dtype, device=device, requires_grad=True)
        log_sigma = torch.full((d,), self.init_log_sigma, dtype=dtype,
                               device=device, requires_grad=True)

        dx_t = torch.as_tensor(dx, dtype=dtype, device=device)
        v_t = torch.as_tensor(v, dtype=dtype, device=device)

        optimizer = torch.optim.Adam([mu, log_sigma], lr=self.learning_rate)

        elbo_trace = np.empty(self.n_steps)
        iterator = range(self.n_steps)
        if self.progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc=f"VI y={y}")
            except ImportError:
                pass

        for step in iterator:
            optimizer.zero_grad()
            elbo = _elbo_estimate(mu, log_sigma, dx_t, v_t, y,
                                  self.n_samples, model, prior)
            loss = -elbo
            loss.backward()
            torch.nn.utils.clip_grad_norm_([mu, log_sigma], max_norm=self.grad_clip)
            optimizer.step()
            elbo_trace[step] = elbo.detach().item()

        with torch.no_grad():
            elbo_final = _elbo_estimate(
                mu, log_sigma, dx_t, v_t, y, self.final_n_samples, model, prior,
            )
            theta_mean_t, _ = prior.transform(mu)
            theta_mean = {k: v_.detach().cpu().numpy() for k, v_ in theta_mean_t.items()}

        return {
            "elbo": float(elbo_final.item()),
            "mu": mu.detach().cpu().numpy(),
            "log_sigma": log_sigma.detach().cpu().numpy(),
            "theta_mean": theta_mean,
            "elbo_trace": elbo_trace,
        }

    # ------------------------------------------------------------------
    # Bayes factor / posterior (Inference interface)
    # ------------------------------------------------------------------

    def run(
        self,
        dx: np.ndarray,
        v: np.ndarray,
        model: Model,
        prior: Prior,
        *,
        pi0: float = 0.5,
        seed: Optional[int] = None,
        record_pi_t: bool = False,
        **_: Any,
    ) -> Dict[str, Any]:
        """Run VI under both outcomes and return approximate posterior."""
        if record_pi_t:
            # VI does not produce an online posterior trace, only a posterior at T.
            raise NotImplementedError(
                "VIInference does not support record_pi_t=True; use SMCInference."
            )

        res0 = self.marginal_likelihood(dx, v, 0, model, prior, seed=seed)
        res1 = self.marginal_likelihood(
            dx, v, 1, model, prior,
            seed=None if seed is None else seed + 1,
        )

        log_BF = res1["elbo"] - res0["elbo"]
        logit_pi0 = math.log(pi0 / (1.0 - pi0))
        log_post_odds = logit_pi0 + log_BF
        posterior = _stable_sigmoid(log_post_odds)

        return {
            "log_BF": float(log_BF),
            "log_m0": float(res0["elbo"]),
            "log_m1": float(res1["elbo"]),
            "posterior": float(posterior),
            "pi0": float(pi0),
            "vi0": res0,
            "vi1": res1,
        }
