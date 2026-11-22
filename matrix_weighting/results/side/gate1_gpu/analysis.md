# Gate 1 analysis (planned rules, docs/plan.md section 8) on results/side/gate1_gpu

## T0.5

- M-self vs S-self: mean Δ +0.042 [-0.042, +0.114] (×1.10; A lower in 33% of pairs) → no decision
- M-self vs U: mean Δ +0.033 [-0.124, +0.157] (×1.08; A lower in 39% of pairs) → no decision
- M-self vs TUR: mean Δ +0.069 [-0.135, +0.218] (×1.17; A lower in 39% of pairs) → no decision
- M-orc vs S-orc: mean Δ +0.052 [-0.016, +0.118] (×1.13; A lower in 47% of pairs) → no decision
- S-self vs U: mean Δ -0.009 [-0.114, +0.090] (×0.98; A lower in 47% of pairs) → no decision
- TUR vs U: mean Δ -0.035 [-0.129, +0.058] (×0.92; A lower in 53% of pairs) → no decision

## T1

- M-self vs S-self: mean Δ -0.283 [-0.931, +0.147] (×0.52; A lower in 36% of pairs) → no decision
- M-self vs U: mean Δ -2.160 [-3.085, -1.188] (×0.01; A lower in 83% of pairs) → **beats**
- M-self vs TUR: mean Δ -2.342 [-3.219, -1.384] (×0.00; A lower in 83% of pairs) → **beats**
- M-orc vs S-orc: mean Δ -1.546 [-2.665, -0.462] (×0.03; A lower in 67% of pairs) → **beats**
- S-self vs U: mean Δ -1.876 [-3.106, -0.532] (×0.01; A lower in 89% of pairs) → **beats**
- TUR vs U: mean Δ +0.182 [-0.113, +0.657] (×1.52; A lower in 19% of pairs) → no decision

## T2

- M-self vs S-self: mean Δ -0.232 [-0.634, +0.147] (×0.59; A lower in 67% of pairs) → no decision
- M-self vs U: mean Δ -0.674 [-1.142, -0.268] (×0.21; A lower in 86% of pairs) → **beats**
- M-self vs TUR: mean Δ -0.692 [-1.182, -0.268] (×0.20; A lower in 83% of pairs) → **beats**
- M-orc vs S-orc: mean Δ +0.170 [-0.104, +0.605] (×1.48; A lower in 69% of pairs) → no decision
- S-self vs U: mean Δ -0.442 [-0.877, -0.067] (×0.36; A lower in 75% of pairs) → **beats**
- TUR vs U: mean Δ +0.018 [-0.011, +0.053] (×1.04; A lower in 58% of pairs) → no decision

**H2: partial** (beats at some horizon: {'S-self': False, 'U': True, 'TUR': True}; worse somewhere: False)

## Descriptive table

| system | T | method | n | median log10 E_rms [IQR] | median log10 E_max | E_max<1 |
|---|---|---|---:|---|---:|---:|
| lorenz63 | T0.5 | M-orc | 36 | -2.93 [-3.06, -2.80] | -2.53 | 36/36 |
| lorenz63 | T0.5 | M-self | 36 | -2.88 [-3.06, -2.75] | -2.47 | 36/36 |
| lorenz63 | T0.5 | S-orc | 36 | -3.02 [-3.19, -2.85] | -2.42 | 36/36 |
| lorenz63 | T0.5 | S-self | 36 | -2.99 [-3.09, -2.83] | -2.37 | 36/36 |
| lorenz63 | T0.5 | TUR | 36 | -3.01 [-3.32, -2.71] | -2.68 | 36/36 |
| lorenz63 | T0.5 | U | 36 | -3.05 [-3.17, -2.82] | -2.66 | 36/36 |
| lorenz63 | T1 | M-orc | 36 | -2.36 [-2.60, 1.26] | -1.87 | 23/36 |
| lorenz63 | T1 | M-self | 36 | -2.29 [-2.45, -1.22] | -1.78 | 27/36 |
| lorenz63 | T1 | S-orc | 36 | 1.27 [0.53, 1.43] | 1.58 | 9/36 |
| lorenz63 | T1 | S-self | 36 | -2.27 [-2.76, 1.23] | -1.73 | 24/36 |
| lorenz63 | T1 | TUR | 36 | 1.29 [1.26, 1.44] | 1.61 | 5/36 |
| lorenz63 | T1 | U | 36 | 1.29 [1.26, 1.43] | 1.61 | 7/36 |
| lorenz63 | T2 | M-orc | 36 | 1.31 [1.17, 1.32] | 1.58 | 3/36 |
| lorenz63 | T2 | M-self | 36 | 1.15 [0.24, 1.29] | 1.50 | 8/36 |
| lorenz63 | T2 | S-orc | 36 | 1.35 [1.14, 1.44] | 1.61 | 6/36 |
| lorenz63 | T2 | S-self | 36 | 1.32 [0.87, 1.37] | 1.59 | 5/36 |
| lorenz63 | T2 | TUR | 36 | 1.38 [1.29, 1.44] | 1.62 | 0/36 |
| lorenz63 | T2 | U | 36 | 1.40 [1.31, 1.44] | 1.62 | 0/36 |
