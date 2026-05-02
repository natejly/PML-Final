"""Forward samplers for synthetic experiments."""

from .burst_mispricing_log_ar_vol_sampler import (
    BurstMispricingLogARVolSampler,
    default_burst_mispricing_theta,
)
from .mispricing_log_ar_vol_sampler import (
    MispricingLogARVolSampler,
    default_mispricing_theta,
)
from .reversal_momentum_burst_log_ar_vol_sampler import (
    ReversalMomentumBurstLogARVolSampler,
    default_reversal_momentum_burst_theta,
)

SAMPLER_REGISTRY = {
    "reversal": MispricingLogARVolSampler,
    "reversal_burst": BurstMispricingLogARVolSampler,
    "reversal_momentum_burst": ReversalMomentumBurstLogARVolSampler,
}

__all__ = [
    "BurstMispricingLogARVolSampler",
    "MispricingLogARVolSampler",
    "ReversalMomentumBurstLogARVolSampler",
    "SAMPLER_REGISTRY",
    "default_burst_mispricing_theta",
    "default_mispricing_theta",
    "default_reversal_momentum_burst_theta",
]
