"""pml_market: Bayesian inverse-problem inference for binary prediction markets.

Implements the Gaussian latent-type model from Example 3.1 of Madrigal-Cianci,
Monsalve Maya, Breakey (2026), with SMC and VI inference and the diagnostics
needed to reproduce the paper's experiments on synthetic and real Polymarket
data.

Public OOP API
--------------

The package is organised around three pluggable interfaces in ``core``:

  * ``Model``     - likelihood for log-odds increments under outcome y.
  * ``Prior``     - prior on theta + bijector to/from R^d.
  * ``Inference`` - engine producing P(Y=1|h) and log Bayes factor.

A concrete inference run::

    from pml_market import (
        InverseProblem, GaussianLatentTypeModel, LatentTypePrior,
        SMCInference, VIInference,
    )

    problem = InverseProblem(GaussianLatentTypeModel(), LatentTypePrior())
    smc = SMCInference(n_particles=1000, mcmc_steps=5)
    res = problem.infer(dx, v, smc, pi0=0.5, seed=0, record_pi_t=True)
"""

from .core import Model, Prior, Inference, InverseProblem
from .model import GaussianLatentTypeModel
from .priors import LatentTypePrior
from .volume_model import VolumeLognormalModel
from .volume_prior import VolumeLognormalPrior, VolumeLognormalEBPrior
from .smc import SMCInference
from .vi import VIInference

# Submodules remain importable for low-level access.
from . import model, priors, synthetic, smc, vi, diagnostics, data  # noqa: F401
from . import volume_model, volume_prior  # noqa: F401

__all__ = [
    # OOP API
    "Model", "Prior", "Inference", "InverseProblem",
    "GaussianLatentTypeModel", "LatentTypePrior",
    "VolumeLognormalModel", "VolumeLognormalPrior", "VolumeLognormalEBPrior",
    "SMCInference", "VIInference",
    # Submodules
    "model", "priors", "synthetic", "smc", "vi", "diagnostics", "data",
    "volume_model", "volume_prior",
]
