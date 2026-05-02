# Synthetic Posterior Concentration Analysis

## Posterior Concentration

| model | n | T | empirical slope delta | final median log error | final q10 | final q90 |
|---|---:|---:|---:|---:|---:|---:|
| reversal | 1000 | 500 | 0.0954 | -48.554 | -64.246 | -35.951 |
| reversal_burst | 1000 | 500 | 0.0742 | -38.130 | -51.330 | -25.717 |
| reversal_momentum_burst | 1000 | 500 | 0.0019 | -1.721 | -9.074 | -0.003 |

## KL Projection

| model | horizon | delta_hat | true ll/T | best wrong ll/T | optimizer success |
|---|---:|---:|---:|---:|---|
| reversal | 10 | 0.0783 | -1.3059 | -1.3842 | False |
| reversal | 25 | 0.0865 | -1.3883 | -1.4748 | False |
| reversal | 50 | 0.0922 | -1.4196 | -1.5118 | False |
| reversal | 100 | 0.0927 | -1.4364 | -1.5291 | False |
| reversal | 200 | 0.0909 | -1.4433 | -1.5342 | False |
| reversal | 500 | 0.0916 | -1.4506 | -1.5422 | False |
| reversal_burst | 10 | 0.0571 | -1.5194 | -1.5764 | False |
| reversal_burst | 25 | 0.0633 | -1.6442 | -1.7074 | False |
| reversal_burst | 50 | 0.0675 | -1.6911 | -1.7587 | False |
| reversal_burst | 100 | 0.0683 | -1.7124 | -1.7807 | False |
| reversal_burst | 200 | 0.0692 | -1.7256 | -1.7948 | False |
| reversal_burst | 500 | 0.0688 | -1.7330 | -1.8017 | False |
| reversal_momentum_burst | 10 | 0.0031 | -1.6117 | -1.6148 | False |
| reversal_momentum_burst | 25 | 0.0045 | -1.7453 | -1.7498 | False |
| reversal_momentum_burst | 50 | 0.0051 | -1.7974 | -1.8025 | False |
| reversal_momentum_burst | 100 | 0.0047 | -1.8178 | -1.8226 | False |
| reversal_momentum_burst | 200 | 0.0054 | -1.8321 | -1.8375 | False |
| reversal_momentum_burst | 500 | 0.0053 | -1.8418 | -1.8470 | False |
