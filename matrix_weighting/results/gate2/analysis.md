# Gate 2 analysis (planned rules, docs/plan.md section 9; Lorenz-part error)

## k = 0

- M-self vs U: mean Δ -2.160 [-3.085, -1.188] (×0.01; A lower in 83% of pairs) → **beats**
- M-self vs S-self: mean Δ -0.283 [-0.931, +0.147] (×0.52; A lower in 36% of pairs) → no decision
- M-self vs TUR: mean Δ -2.342 [-3.219, -1.384] (×0.00; A lower in 83% of pairs) → **beats**
- M-orc vs S-orc: mean Δ -1.546 [-2.665, -0.462] (×0.03; A lower in 67% of pairs) → **beats**

## k = 3

- M-self vs U: mean Δ -2.166 [-2.925, -1.374] (×0.01; A lower in 100% of pairs) → **beats**
- M-self vs S-self: mean Δ -0.886 [-1.352, -0.418] (×0.13; A lower in 94% of pairs) → **beats**
- M-self vs TUR: mean Δ -2.455 [-3.147, -1.687] (×0.00; A lower in 100% of pairs) → **beats**
- M-orc vs S-orc: mean Δ -1.257 [-2.151, -0.480] (×0.06; A lower in 83% of pairs) → **beats**

## k = 9

- M-self vs U: mean Δ -2.447 [-3.132, -1.706] (×0.00; A lower in 97% of pairs) → **beats**
- M-self vs S-self: mean Δ -1.687 [-2.306, -1.090] (×0.02; A lower in 97% of pairs) → **beats**
- M-self vs TUR: mean Δ -2.642 [-3.269, -1.914] (×0.00; A lower in 97% of pairs) → **beats**
- M-orc vs S-orc: mean Δ -0.634 [-1.280, -0.048] (×0.23; A lower in 83% of pairs) → **beats**

## k = 29

- M-self vs U: mean Δ -2.437 [-3.042, -1.719] (×0.00; A lower in 94% of pairs) → **beats**
- M-self vs S-self: mean Δ -2.010 [-2.593, -1.387] (×0.01; A lower in 94% of pairs) → **beats**
- M-self vs TUR: mean Δ -2.497 [-3.121, -1.768] (×0.00; A lower in 94% of pairs) → **beats**
- M-orc vs S-orc: mean Δ -0.286 [-0.597, -0.024] (×0.52; A lower in 36% of pairs) → **beats**

**H3: not supported** (gain at k=29 minus gain at k=0: +0.277 [-0.303, +0.981])

## Descriptive table (Lorenz-part error)

| system | T | method | n | median log10 E_rms [IQR] | median log10 E_max | E_max<1 |
|---|---|---|---:|---|---:|---:|
| lorenz63 | T1 | M-orc | 36 | -2.36 [-2.60, 1.26] | -1.87 | 23/36 |
| lorenz63 | T1 | M-self | 36 | -2.29 [-2.45, -1.22] | -1.78 | 27/36 |
| lorenz63 | T1 | S-orc | 36 | 1.27 [0.53, 1.43] | 1.58 | 9/36 |
| lorenz63 | T1 | S-self | 36 | -2.27 [-2.76, 1.23] | -1.73 | 24/36 |
| lorenz63 | T1 | TUR | 36 | 1.29 [1.26, 1.44] | 1.61 | 5/36 |
| lorenz63 | T1 | U | 36 | 1.29 [1.26, 1.43] | 1.61 | 7/36 |
| lorenz63_damped29 | T1 | M-orc | 36 | 1.27 [1.26, 1.44] | 1.60 | 5/36 |
| lorenz63_damped29 | T1 | M-self | 36 | -1.73 [-1.90, -0.92] | -1.19 | 27/36 |
| lorenz63_damped29 | T1 | S-orc | 36 | 1.27 [1.26, 1.36] | 1.58 | 3/36 |
| lorenz63_damped29 | T1 | S-self | 36 | 1.18 [0.59, 1.26] | 1.49 | 5/36 |
| lorenz63_damped29 | T1 | TUR | 36 | 1.28 [1.26, 1.43] | 1.61 | 0/36 |
| lorenz63_damped29 | T1 | U | 36 | 1.28 [1.26, 1.43] | 1.61 | 0/36 |
| lorenz63_damped3 | T1 | M-orc | 36 | 1.00 [-2.20, 1.29] | 1.37 | 17/36 |
| lorenz63_damped3 | T1 | M-self | 36 | -1.99 [-2.25, -0.15] | -1.52 | 27/36 |
| lorenz63_damped3 | T1 | S-orc | 36 | 1.28 [1.26, 1.43] | 1.61 | 5/36 |
| lorenz63_damped3 | T1 | S-self | 36 | -0.69 [-1.77, 1.26] | -0.25 | 20/36 |
| lorenz63_damped3 | T1 | TUR | 36 | 1.30 [1.26, 1.44] | 1.61 | 4/36 |
| lorenz63_damped3 | T1 | U | 36 | 1.29 [1.26, 1.43] | 1.61 | 7/36 |
| lorenz63_damped9 | T1 | M-orc | 36 | 1.26 [-1.44, 1.38] | 1.55 | 10/36 |
| lorenz63_damped9 | T1 | M-self | 36 | -1.93 [-2.02, -0.95] | -1.39 | 27/36 |
| lorenz63_damped9 | T1 | S-orc | 36 | 1.28 [1.26, 1.43] | 1.61 | 4/36 |
| lorenz63_damped9 | T1 | S-self | 36 | 1.08 [-0.25, 1.26] | 1.38 | 7/36 |
| lorenz63_damped9 | T1 | TUR | 36 | 1.29 [1.26, 1.44] | 1.61 | 0/36 |
| lorenz63_damped9 | T1 | U | 36 | 1.29 [1.26, 1.43] | 1.61 | 3/36 |
