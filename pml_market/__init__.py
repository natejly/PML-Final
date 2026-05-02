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
        InverseProblem, BaseModel, BasePrior,
        SMCInference, VIInference,
    )

    problem = InverseProblem(BaseModel(), BasePrior())
    smc = SMCInference(n_particles=1000, mcmc_steps=5)
    res = problem.infer(dx, v, smc, pi0=0.5, seed=0, record_pi_t=True)
"""

from .core import Model, Prior, Inference, InverseProblem
from .models.base_model import BaseModel
from .priors.base_prior import BasePrior
from .models.burst_mispricing_log_ar_vol_model import BurstMispricingLogARVolModel
from .priors.burst_mispricing_log_ar_vol_prior import BurstMispricingLogARVolPrior
from .models.gated_reversal_momentum_burst_log_ar_vol_model import (
    GatedReversalMomentumBurstLogARVolModel,
)
from .priors.gated_reversal_momentum_burst_log_ar_vol_prior import (
    GatedReversalMomentumBurstLogARVolPrior,
)
from .models.gaussian_vol_model import GaussianVolModel
from .priors.gaussian_vol_prior import GaussianVolPrior
from .models.log_ar_vol_model import LogARVolModel
from .priors.log_ar_vol_prior import LogARVolPrior
from .models.mispricing_log_ar_vol_model import MispricingLogARVolModel
from .priors.mispricing_log_ar_vol_prior import MispricingLogARVolPrior
from .models.reversal_momentum_burst_log_ar_vol_model import (
    ReversalMomentumBurstLogARVolModel,
)
from .priors.reversal_momentum_burst_log_ar_vol_prior import (
    ReversalMomentumBurstLogARVolPrior,
)
from .samplers import (
    BurstMispricingLogARVolSampler,
    GatedReversalMomentumBurstLogARVolSampler,
    MispricingLogARVolSampler,
    ReversalMomentumBurstLogARVolSampler,
)
from .inference.smc import SMCInference
from .inference.vi import VIInference

# Submodules remain importable for low-level access.
from .models import (
    base_model,
    burst_mispricing_log_ar_vol_model,
    gated_reversal_momentum_burst_log_ar_vol_model,
    gaussian_vol_model,
    log_ar_vol_model,
    mispricing_log_ar_vol_model,
    reversal_momentum_burst_log_ar_vol_model,
)
from .priors import (
    base_prior,
    burst_mispricing_log_ar_vol_prior,
    gated_reversal_momentum_burst_log_ar_vol_prior,
    gaussian_vol_prior,
    log_ar_vol_prior,
    mispricing_log_ar_vol_prior,
    reversal_momentum_burst_log_ar_vol_prior,
)
from . import synthetic, diagnostics, data, samplers
from .inference import smc, vi  # noqa: F401

__all__ = [
    # OOP API
    "Model", "Prior", "Inference", "InverseProblem",
    "BaseModel", "BasePrior",
    "BurstMispricingLogARVolModel", "BurstMispricingLogARVolPrior",
    "GatedReversalMomentumBurstLogARVolModel", "GatedReversalMomentumBurstLogARVolPrior",
    "GaussianVolModel", "GaussianVolPrior",
    "LogARVolModel", "LogARVolPrior",
    "MispricingLogARVolModel", "MispricingLogARVolPrior",
    "ReversalMomentumBurstLogARVolModel", "ReversalMomentumBurstLogARVolPrior",
    "BurstMispricingLogARVolSampler",
    "GatedReversalMomentumBurstLogARVolSampler",
    "MispricingLogARVolSampler",
    "ReversalMomentumBurstLogARVolSampler",
    "SMCInference", "VIInference",
    # Submodules
    "base_model", "base_prior", "synthetic", "samplers",
    "smc", "vi", "diagnostics", "data",
    "burst_mispricing_log_ar_vol_model", "burst_mispricing_log_ar_vol_prior",
    "gated_reversal_momentum_burst_log_ar_vol_model",
    "gated_reversal_momentum_burst_log_ar_vol_prior",
    "gaussian_vol_model", "gaussian_vol_prior",
    "log_ar_vol_model", "log_ar_vol_prior",
    "mispricing_log_ar_vol_model", "mispricing_log_ar_vol_prior",
    "reversal_momentum_burst_log_ar_vol_model",
    "reversal_momentum_burst_log_ar_vol_prior",
]
