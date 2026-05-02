# Synthetic Posterior Concentration Analysis

## Posterior Concentration

| model | n | T | empirical slope delta | final median log error | final q10 | final q90 |
|---|---:|---:|---:|---:|---:|---:|
| gated_reversal_momentum_burst | 1000 | 500 | 0.0049 | -3.156 | -10.625 | -0.060 |

## KL Projection

| model | horizon | delta_hat | true ll/T | best wrong ll/T | optimizer success |
|---|---:|---:|---:|---:|---|
| gated_reversal_momentum_burst | 10 | 0.0097 | -1.5241 | -1.5339 | False |
| gated_reversal_momentum_burst | 25 | 0.0144 | -1.6322 | -1.6466 | False |
| gated_reversal_momentum_burst | 50 | 0.0140 | -1.6724 | -1.6864 | False |
| gated_reversal_momentum_burst | 100 | 0.0135 | -1.6868 | -1.7002 | False |
| gated_reversal_momentum_burst | 200 | 0.0129 | -1.6924 | -1.7053 | False |
| gated_reversal_momentum_burst | 500 | 0.0126 | -1.6956 | -1.7082 | False |
