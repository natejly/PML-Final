"""pml_market: Bayesian inverse-problem inference for binary prediction markets.

Implements the Gaussian latent-type model from Example 3.1 of Madrigal-Cianci,
Monsalve Maya, Breakey (2026), with SMC and VI inference and the diagnostics
needed to reproduce the paper's experiments on synthetic and real Polymarket
data.

Submodules are imported lazily so partial installs (e.g. without torch) still
let you load the rest of the package.
"""

__all__ = ["model", "priors", "synthetic", "smc", "vi", "diagnostics", "data"]
