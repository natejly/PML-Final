# PML-Final

Implementation of the Bayesian inverse-problem framework for prediction markets
from Madrigal-Cianci, Monsalve Maya, and Breakey (2026), "Prediction Markets as
Bayesian Inverse Problems," with experiments on real Polymarket data.

## Layout

- `pml_market/` — model, priors, SMC, VI, diagnostics, data adapters.
- `notebooks/polymarket_data_pull.ipynb` — pulls binary markets via the public
  Polymarket Gamma/Data/CLOB APIs (stdlib only).
- `notebooks/01_synthetic_sanity.ipynb` — reproduces the four synthetic
  experiments from the paper as a correctness gate.
- `notebooks/02_single_market_deepdive.ipynb` — full diagnostic suite on one or
  two resolved Polymarket binaries.
- `notebooks/03_panel_evaluation.ipynb` — VI run over a panel of resolved
  binary markets with calibration, Brier, and IG aggregates.

## Install

```bash
pip install -r requirements.txt
```

## Quick start

The package is organised around three pluggable interfaces (`Model`,
`Prior`, `Inference`) wired together by an `InverseProblem` facade. To run
inference, choose one of each and call `problem.infer(...)`:

```python
from pml_market.synthetic import simulate_history
from pml_market import (
    InverseProblem, GaussianLatentTypeModel, LatentTypePrior,
    SMCInference, VIInference,
)

dx, v, y = simulate_history(T=200, y_true=1, seed=0)

problem = InverseProblem(GaussianLatentTypeModel(), LatentTypePrior())
smc = SMCInference(n_particles=500, mcmc_steps=3)
res = problem.infer(dx, v, smc, pi0=0.5, seed=0, record_pi_t=True)
print("posterior P(Y=1|H)=", res["posterior"], " truth=", y)

# Swap in mean-field VI on the same problem:
vi = VIInference(n_steps=1000, n_samples=8)
res_vi = problem.infer(dx, v, vi, pi0=0.5, seed=0)
```

To plug in an alternative model, prior, or inference engine, subclass the
matching ABC in `pml_market.core` and pass the new instance to
`InverseProblem` / `problem.infer`.

## Modeling volume endogenously

The default `GaussianLatentTypeModel` only specifies the conditional
likelihood

\[
p(\Delta x_{1:T} \mid v_{1:T},\, Y, \theta),
\]

treating the trade-volume sequence as exogenous. If volume should also
carry signal about `Y`, factor the joint distribution as

\[
p(\Delta x_{1:T},\, v_{1:T} \mid Y = y, \theta)
  \;=\; p(\Delta x_{1:T} \mid v_{1:T},\, Y = y, \theta_{\text{inc}})
  \;\cdot\; p(v_{1:T} \mid Y = y, \theta_{\text{vol}}),
\]

and keep the original increment factor unchanged.

`pml_market.volume_model.VolumeLognormalModel` and
`pml_market.volume_prior.VolumeLognormalPrior` implement a first-pass
parametric form for the second factor:

\[
\log(1 + v_t) \;\sim\; \mathcal{N}\!\bigl(\mu_v[Y],\; \sigma_v[Y]^2\bigr),
\quad t = 1,\ldots,T \text{ iid},
\]

with `mu_v[y] ~ N(0, 10^2)` and `sigma_v[y] ~ HalfNormal(5)` for
`y in {0, 1}`. The wrapper composes on any inner increment Model, so the
mixture-likelihood part stays plug-replaceable. The Bayes factor produced
by SMC/VI is then over the *joint* observation, picking up Y-dependent
asymmetries in trading activity in addition to the price-direction signal:

```python
from pml_market import (
    InverseProblem, VolumeLognormalModel, VolumeLognormalPrior,
    SMCInference,
)

problem = InverseProblem(VolumeLognormalModel(), VolumeLognormalPrior())
smc = SMCInference(n_particles=500, mcmc_steps=3)
res = problem.infer(dx, v, smc, pi0=0.5, seed=0)
print("joint posterior P(Y=1|H)=", res["posterior"])
```

Both the volume model and the volume prior accept a base instance, so you
can mix and match — e.g. swap the inner increment model while keeping the
lognormal volume term, or attach a different volume process by subclassing
`Model` / `Prior` directly. To replace the iid lognormal with a
heavier-tailed or autocorrelated volume process, copy
`pml_market/volume_model.py` and `pml_market/volume_prior.py` as the
starting point and edit `_volume_logpdf` plus the four extra prior terms.

Notebook `02_single_market_deepdive.ipynb` runs both models on the same
resolved Polymarket market and reports the difference in log Bayes
factor, posterior, and information gain so you can see how much the
volume term contributes on real data.
