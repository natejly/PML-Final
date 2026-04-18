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

```python
from pml_market.synthetic import simulate_history
from pml_market.smc import bayes_factor_smc

dx, v, y = simulate_history(T=200, y_true=1, seed=0)
res = bayes_factor_smc(dx, v, n_particles=500, seed=0)
print("posterior P(Y=1|H)=", res["posterior"], " truth=", y)
```
