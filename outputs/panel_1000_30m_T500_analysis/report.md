# Panel SMC Post-Processing Report

Input: `outputs/panel_1000_30m_T500_p1500_m2_all.pkl`

## Model Summary

| model | n | mean log loss | median log loss | mean Brier | accuracy | mean P(true) | overconf wrong 0.9 | ECE | mean runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| market_final | 1000 | 0.001 | 0.001 | 0.000 | 1.000 | 0.999 | 0.000 | 0.001 |  |
| market_pre_final | 1000 | 0.002 | 0.001 | 0.000 | 1.000 | 0.998 | 0.000 | 0.002 |  |
| base | 1000 | 6.789 | 1.852 | 0.519 | 0.444 | 0.448 | 0.444 | 0.522 | 10.62 |
| log_ar_volume | 1000 | 12.160 | 1.483 | 0.498 | 0.497 | 0.497 | 0.483 | 0.500 | 28.89 |
| reversal_momentum_burst | 1000 | 12.591 | 0.039 | 0.493 | 0.506 | 0.506 | 0.492 | 0.494 | 18.41 |
| reversal | 1000 | 12.739 | 4.333 | 0.522 | 0.472 | 0.471 | 0.511 | 0.523 | 29.30 |
| reversal_burst | 1000 | 13.301 | 5.777 | 0.512 | 0.486 | 0.486 | 0.506 | 0.512 | 17.35 |
| rw_volume | 1000 | 14.404 | 27.631 | 0.522 | 0.477 | 0.477 | 0.522 | 0.523 | 37.56 |

Lower log loss, lower Brier, lower ECE, and lower overconfident-wrong rate are better. Higher accuracy and P(true) are better.

## Compared To `base`

| model | n | delta log loss | log loss win rate | delta Brier | Brier win rate | mean abs posterior diff | disagreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| market_final | 1000 | -6.788 | 0.726 | -0.519 | 0.726 | 0.551 | 0.556 |
| market_pre_final | 1000 | -6.788 | 0.726 | -0.519 | 0.726 | 0.551 | 0.556 |
| log_ar_volume | 1000 | 5.371 | 0.449 | -0.021 | 0.449 | 0.473 | 0.475 |
| reversal_momentum_burst | 1000 | 5.802 | 0.449 | -0.026 | 0.449 | 0.496 | 0.496 |
| reversal | 1000 | 5.950 | 0.427 | 0.003 | 0.427 | 0.485 | 0.492 |
| reversal_burst | 1000 | 6.512 | 0.436 | -0.007 | 0.436 | 0.515 | 0.522 |
| rw_volume | 1000 | 7.615 | 0.411 | 0.003 | 0.411 | 0.495 | 0.495 |

Negative deltas mean the candidate beat the reference on average.

## Posterior Trace Diagnostics

| model | n | mean max step change | mean decision flips | mean frac time conf >= 0.9 |
|---|---:|---:|---:|---:|
| base | 1000 | 0.456 | 3.565 | 0.638 |
| log_ar_volume | 1000 | 0.719 | 3.173 | 0.914 |
| reversal | 1000 | 0.735 | 3.354 | 0.909 |
| reversal_burst | 1000 | 0.553 | 3.032 | 0.963 |
| reversal_momentum_burst | 1000 | 0.511 | 2.739 | 0.961 |
| rw_volume | 1000 | 0.750 | 1.559 | 0.984 |

## Output Files

- `model_summary.csv`: aggregate scoring table.
- `model_results_long.csv`: one row per market-model result.
- `model_results_wide.csv`: one row per market with per-model columns.
- `vs_reference.csv`: per-model deltas against the selected reference.
- `pairwise_model_comparisons.csv`: posterior disagreement and metric deltas for all model pairs.
- `calibration_bins.csv`: calibration-bin data for reliability plots.
- `hardest_markets.csv`: markets where the model family struggled most.
- `trace_features.csv` and `trace_summary.csv`: written only when posterior traces are present.
