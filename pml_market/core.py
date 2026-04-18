"""Abstract base classes and the InverseProblem facade.

The package is organised around three pluggable interfaces:

  * Model      : likelihood for log-odds increments under outcome y in {0, 1}.
  * Prior      : prior on the model parameters + bijector to/from R^d.
  * Inference  : engine producing the posterior P(Y=1 | h) and log Bayes factor.

A concrete inference run consists of choosing one of each, e.g.::

    from pml_market import (GaussianLatentTypeModel, LatentTypePrior,
                            SMCInference, InverseProblem)

    problem = InverseProblem(GaussianLatentTypeModel(), LatentTypePrior())
    smc = SMCInference(n_particles=1000, mcmc_steps=5)
    res = problem.infer(dx, v, smc, pi0=0.5, seed=0, record_pi_t=True)

To plug in a new model you subclass `Model` and implement `mixture_logpdf`.
For a new prior subclass `Prior` and implement the bijector pair plus
`log_prior_unconstrained`.  For a new inference algorithm subclass
`Inference` and implement `run`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Model(ABC):
    """Likelihood for log-odds increments under a binary outcome y in {0, 1}.

    Subclasses must declare `PARAM_NAMES` and implement `mixture_logpdf`.
    Backend dispatch (numpy / torch) is the responsibility of the subclass:
    methods should accept either array type and return the same kind.
    """

    PARAM_NAMES: Tuple[str, ...] = ()

    @abstractmethod
    def mixture_logpdf(self, dx, v, y: int, theta: Mapping[str, Any]):
        """Per-step log f_y(dx_t | v_t, theta).  Returns shape (..., T)."""

    def loglik(self, dx, v, y: int, theta: Mapping[str, Any]):
        """Total log-likelihood sum_t log f_y(dx_t | v_t, theta).

        Default implementation sums `mixture_logpdf` over the time axis.
        Override for per-model speedups.
        """
        return self.mixture_logpdf(dx, v, y, theta).sum(axis=-1)


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------


class Prior(ABC):
    """Prior on theta together with a bijector to/from unconstrained R^d.

    The bijector is required for any inference algorithm that needs gradients
    or random-walk MCMC (i.e. SMC and VI).  `log_prior_unconstrained(z)`
    must equal `log_prior(theta(z)) + log|det dtheta/dz|` and be
    differentiable when `z` is a torch tensor.
    """

    UNCONSTRAINED_DIM: int = 0

    @abstractmethod
    def sample(self, rng: np.random.Generator, n: int = 1) -> Mapping[str, np.ndarray]:
        """Draw n iid theta samples from the prior.  Returns batched dict."""

    @abstractmethod
    def transform(self, z) -> Tuple[Mapping[str, Any], Any]:
        """Map z (..., d) -> (theta, log|det dtheta/dz|).  Differentiable in torch."""

    @abstractmethod
    def to_unconstrained(self, theta: Mapping[str, Any]) -> np.ndarray:
        """Inverse of `transform`: theta dict -> unconstrained vector(s)."""

    @abstractmethod
    def log_prior_unconstrained(self, z):
        """log pi(z) = log Pi(theta(z)) + log|det dtheta/dz|.  Differentiable."""

    def log_prior(self, theta: Mapping[str, Any]) -> float:
        """log Pi(theta) on the constrained space.

        Optional; subclasses override if needed.  The default raises so callers
        cannot silently get a wrong value.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.log_prior is not implemented; subclass "
            "should override if constrained-space prior densities are needed."
        )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


class Inference(ABC):
    """Inference engine: dx, v, Model, Prior -> posterior P(Y=1|h) and log BF.

    Concrete subclasses (`SMCInference`, `VIInference`) hold algorithm
    hyperparameters in their constructor and perform a single inference run
    via `.run(...)`.
    """

    @abstractmethod
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
        **kwargs: Any,
    ) -> dict:
        """Run inference on a single history.

        Returns
        -------
        dict
            Must contain at minimum:
            ``log_BF``, ``log_m0``, ``log_m1``, ``posterior``, ``pi0``.
            If ``record_pi_t=True`` should additionally return:
            ``pi_t`` (T,), ``log_BF_t`` (T,).
        """


# ---------------------------------------------------------------------------
# InverseProblem facade
# ---------------------------------------------------------------------------


class InverseProblem:
    """Bundles a (Model, Prior) pair so the same inverse problem can be
    re-used across many histories or inference algorithms."""

    def __init__(self, model: Model, prior: Prior):
        self.model = model
        self.prior = prior

    def __repr__(self) -> str:
        return (f"InverseProblem(model={type(self.model).__name__}, "
                f"prior={type(self.prior).__name__})")

    def infer(
        self,
        dx: np.ndarray,
        v: np.ndarray,
        inference: Inference,
        *,
        pi0: float = 0.5,
        seed: Optional[int] = None,
        record_pi_t: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Run `inference` on `(dx, v)` under this problem's (Model, Prior)."""
        return inference.run(
            dx, v, self.model, self.prior,
            pi0=pi0, seed=seed, record_pi_t=record_pi_t, **kwargs,
        )
