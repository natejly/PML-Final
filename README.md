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
3. Implemented 4 new iterations of volume modeling, doing baseline testing on the 2 override markets. Tried log autoregressive (AR) model (would be symmetric in Y, so non-identifiable). This was the only one tried under volume-only-Markov. Then, proceeded with joint Markov model, trying 1. mispricing log AR model 2. mispricing + burst (non-Y dependent) AR model 3. mispricing + burst + momentum AR model. Reversal/mispricing means that if the increment in the previous period was in the opposite direction of the outcome, then volume increases due to mispricing opportunity. momentum means that if the increment in the previous period is in direction of outcome, then volume increases due to traders piling onto the markets (e.g. news getting absorbed over time).
4. Small changes to gitignore (previous commit relative to this one)
5. Change to base_prior log for numerical stability and added time bar for data pulling
6. Added main evaluation script. includes runtime flags for parallelization, number of markets, etc. After 30Apr26 run finished, added an evaluation script. Currently WIP, need to reduce to necessary set of results.
7. Created samplers for the 3 jointly Markov models, and performed experiment 1 from the paper in analysis script. Used T=500, 1000 samples, and did inference with 1500 particles. Computed an estimated outcome separation gap by using the expected value under the true forward law of the log likliehood difference. The expectation was calculated using the empirical distribution from a fresh draw. Got results from analyze_synthetic_posterior_concentration.py script. Found that the most flexible model class (burst + momentum + reversal) enlarges the nuisance, thus giving wrong-outcome model more ways to imitate true-outcome histories (NOTE: may have a symmetry).
8. Fixed a symmetry in the momentum + reversal + burst model that stemmed from momentum and reversal looking symmetric under different outcomes (new gated mom + rev + burst). Added new likelihood model class, prior class, and sampler class for this model. Ran posterior concentration test and confirmed slightly better than original model with symmetry. Still need to run new empirical market evaluation tests. 


**Issue Log:**

1. There is a potential issue where the volume is not strictly positive in the forward process. If you choose a regime of parameters in the forward model s.t. the mean is sufficiently away from 0, this does not really matter. But, this may negatively affect the quality of inference as in real markets, there may be periods of 0 volume (or at least the data we pull says 0 volume)

2. There is a symmetry in the burst + momentum + reversal model. Reversal occurs when the price movement was in the wrong direction in the previous period, causing volume to increase in the next period due to mispricing (e.g. for informed traders). Momentum occurs when the price movement was in the correct direction in the previous period, causing volume to increase in the next period (e.g. uninformed traders follow informed traders, slow to fully price information). BUT there can be a symmetry by sign flip: trending towards truth under Y = 1 can be re-explained with our nuisance structure by reversal after mispricing if Y = 0. 

3. Hypothesis: probably need more particles for the more complex models. rev + mom + burst model vs. gated rev + mom + burst model goes from 8 to 10 params. The number of particles probably scales at least linearly with the number of params. Also, bias-variance tradeoff between identifiability when considering the true market to function as your forward model vs. using a model class to do inference over real market data. 

#Todo: 
2a. Create an asymmetry that is economically explainable in the burst + momentum + reversal model
2b. Redo market analysis for this model. Need to update parallelization structure for just 1 model, parallelizing across markets.
2c. Redo posterior concentration analysis for this model

#Todo: 
Do code review. Review SMC implementation, make sure time-dependencies. Understand what the bijector functionality in the priors are. Understand exactly how the gating is done in the gated reversal + momentum model.
