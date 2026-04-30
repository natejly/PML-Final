# PML-Final

Implementation of the Bayesian inverse-problem framework for prediction markets
from Madrigal-Cianci, Monsalve Maya, and Breakey (2026), "Prediction Markets as
Bayesian Inverse Problems," with experiments on real Polymarket data.

## Layout

- `pml_market/` — model, priors, SMC, VI, diagnostics, data adapters.
- `notebooks/polymarket_data_pull.ipynb` — pulls binary markets via the public
  Polymarket Gamma/Data/CLOB APIs (stdlib only).
- `notebooks/1_paper_replicate.ipynb` — reproduces the four synthetic
  experiments from the paper as a correctness gate.
- `notebooks/2_single_market_test_FG.ipynb` — full diagnostic suite on one
  resolved Polymarket binary with the current Gaussian-volume model.
- `notebooks/2_single_market_test_NL.ipynb` — earlier single-market notebook
  using the older iid lognormal volume experiment.
- `notebooks/3_panel_evaluation.ipynb` — VI run over a panel of resolved
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
    InverseProblem, BaseModel, BasePrior,
    SMCInference, VIInference,
)

dx, v, y = simulate_history(T=200, y_true=1, seed=0)

problem = InverseProblem(BaseModel(), BasePrior())
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

The default `BaseModel` only specifies the conditional
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

`GaussianVolModel` and `GaussianVolPrior` implement the current Markov
volume extension:

\[
v_t \mid v_{t-1}, \sigma_v
  \;\sim\; \mathcal{N}\!\bigl(v_{t-1},\; \sigma_v^2\bigr),
\quad t = 2,\ldots,T,
\]

with `sigma_v ~ HalfNormal(1)`. By default the first observed volume
contributes no likelihood term, so the joint factor is the base increment
likelihood plus the transition product over `t >= 2`. The Bayes factor
produced by SMC/VI is then over the *joint* observation, picking up
autocorrelated volume dynamics in addition to the price-direction signal:

```python
from pml_market import (
    InverseProblem, GaussianVolModel, GaussianVolPrior,
    SMCInference,
)

problem = InverseProblem(GaussianVolModel(), GaussianVolPrior())
smc = SMCInference(n_particles=500, mcmc_steps=3)
res = problem.infer(dx, v, smc, pi0=0.5, seed=0)
print("joint posterior P(Y=1|H)=", res["posterior"])
```

To attach a different volume process, subclass `Model` / `Prior` directly
or add a new implementation under `pml_market/models` and
`pml_market/priors`.

Notebook `2_single_market_test_FG.ipynb` runs the base and Gaussian-volume
models on the same resolved Polymarket market and reports the difference
in log Bayes factor, posterior, and information gain so you can see how
much the volume term contributes on real data.


**Change Trace (Per-Commit):**

1. Removed legacy names for base model
2. Updated SMC: requires model to expose an incremental_log_pdf method for step 4 of SMC. Difference is, takes pointer to the whole array and a specific time (instead of pre-subsetting in SMC and passing in just that point-in time)

