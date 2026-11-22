# Gate 3 analysis (plan amendment 1; H2 rule per system)

## rossler T5

- M-self vs S-self: mean Δ -0.441 [-1.071, +0.030] (×0.36; A lower in 69% of pairs) → no decision
- M-self vs U: mean Δ -0.767 [-1.364, -0.280] (×0.17; A lower in 69% of pairs) → **beats**
- M-self vs TUR: mean Δ -0.792 [-1.385, -0.307] (×0.16; A lower in 69% of pairs) → **beats**
- D-self vs U: mean Δ -0.633 [-1.077, -0.238] (×0.23; A lower in 78% of pairs) → **beats**
- M-self vs D-self: mean Δ -0.134 [-0.410, +0.113] (×0.73; A lower in 56% of pairs) → no decision

## rossler T10

- M-self vs S-self: mean Δ -1.405 [-1.790, -1.066] (×0.04; A lower in 94% of pairs) → **beats**
- M-self vs U: mean Δ -1.570 [-1.905, -1.262] (×0.03; A lower in 94% of pairs) → **beats**
- M-self vs TUR: mean Δ -1.646 [-1.927, -1.363] (×0.02; A lower in 94% of pairs) → **beats**
- D-self vs U: mean Δ -0.830 [-1.138, -0.555] (×0.15; A lower in 97% of pairs) → **beats**
- M-self vs D-self: mean Δ -0.740 [-1.054, -0.454] (×0.18; A lower in 94% of pairs) → **beats**

**H4 (rossler): supported** (beats at some window: {'S-self': True, 'U': True, 'TUR': True}; worse somewhere: False)

## lorenz96_n8 T1

- M-self vs S-self: mean Δ -0.390 [-0.526, -0.234] (×0.41; A lower in 86% of pairs) → **beats**
- M-self vs U: mean Δ -1.153 [-1.373, -0.961] (×0.07; A lower in 100% of pairs) → **beats**
- M-self vs TUR: mean Δ -1.791 [-2.038, -1.536] (×0.02; A lower in 100% of pairs) → **beats**
- D-self vs U: mean Δ -0.484 [-0.725, -0.211] (×0.33; A lower in 78% of pairs) → **beats**
- M-self vs D-self: mean Δ -0.670 [-0.862, -0.493] (×0.21; A lower in 100% of pairs) → **beats**

## lorenz96_n8 T2

- M-self vs S-self: mean Δ -0.377 [-0.774, +0.003] (×0.42; A lower in 75% of pairs) → no decision
- M-self vs U: mean Δ -0.732 [-1.141, -0.377] (×0.19; A lower in 92% of pairs) → **beats**
- M-self vs TUR: mean Δ -0.796 [-1.207, -0.432] (×0.16; A lower in 94% of pairs) → **beats**
- D-self vs U: mean Δ -0.116 [-0.232, -0.002] (×0.77; A lower in 69% of pairs) → no decision
- M-self vs D-self: mean Δ -0.616 [-0.960, -0.310] (×0.24; A lower in 92% of pairs) → **beats**

**H4 (lorenz96_n8): supported** (beats at some window: {'S-self': True, 'U': True, 'TUR': True}; worse somewhere: False)

## Descriptive table

| system | T | method | n | median log10 E_rms [IQR] | median log10 E_max | E_max<1 |
|---|---|---|---:|---|---:|---:|
| lorenz96_n8 | T1 | D-self | 36 | -1.32 [-1.74, -0.92] | -1.08 | 35/36 |
| lorenz96_n8 | T1 | M-self | 36 | -1.95 [-2.14, -1.88] | -1.63 | 36/36 |
| lorenz96_n8 | T1 | S-self | 36 | -1.63 [-1.87, -1.27] | -1.20 | 36/36 |
| lorenz96_n8 | T1 | TUR | 36 | 0.02 [-0.70, 0.32] | 0.39 | 13/36 |
| lorenz96_n8 | T1 | U | 36 | -0.97 [-1.23, -0.54] | -0.70 | 29/36 |
| lorenz96_n8 | T2 | D-self | 36 | 1.05 [0.96, 1.10] | 1.22 | 0/36 |
| lorenz96_n8 | T2 | M-self | 36 | 0.63 [-0.32, 0.90] | 0.94 | 10/36 |
| lorenz96_n8 | T2 | S-self | 36 | 0.90 [0.50, 1.01] | 1.16 | 1/36 |
| lorenz96_n8 | T2 | TUR | 36 | 1.12 [1.10, 1.14] | 1.28 | 0/36 |
| lorenz96_n8 | T2 | U | 36 | 1.09 [1.04, 1.13] | 1.27 | 0/36 |
| rossler | T10 | D-self | 36 | -0.65 [-0.92, -0.05] | -0.14 | 20/36 |
| rossler | T10 | M-self | 36 | -1.50 [-1.61, -1.14] | -1.05 | 30/36 |
| rossler | T10 | S-self | 36 | 0.36 [-0.56, 1.01] | 0.80 | 13/36 |
| rossler | T10 | TUR | 36 | 0.55 [0.12, 0.80] | 0.89 | 1/36 |
| rossler | T10 | U | 36 | 0.29 [0.03, 0.94] | 0.65 | 4/36 |
| rossler | T5 | D-self | 36 | -1.73 [-2.40, 0.71] | -1.39 | 26/36 |
| rossler | T5 | M-self | 36 | -2.38 [-2.60, 1.16] | -2.03 | 25/36 |
| rossler | T5 | S-self | 36 | -2.03 [-2.25, 0.88] | -1.60 | 21/36 |
| rossler | T5 | TUR | 36 | -0.77 [-1.99, 0.93] | -0.55 | 19/36 |
| rossler | T5 | U | 36 | -0.79 [-1.89, 0.89] | -0.60 | 18/36 |
