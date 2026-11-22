# Gate 0 analysis (planned rules, docs/plan.md section 7)

## saddle

- **saddle_k1 (dial 2)**
  - M-orc vs S-orc: mean Δ +0.281 [+0.212, +0.352] (×1.91; A lower in 4% of pairs) → worse
  - M-orc vs U: mean Δ +0.068 [+0.014, +0.135] (×1.17; A lower in 42% of pairs) → worse
  - M-orc vs TUR: mean Δ +0.172 [+0.116, +0.230] (×1.49; A lower in 29% of pairs) → worse
  - S-orc vs U: mean Δ -0.213 [-0.289, -0.148] (×0.61; A lower in 79% of pairs) → **beats**
- **saddle_k4 (dial 5)**
  - M-orc vs S-orc: mean Δ +0.485 [+0.322, +0.666] (×3.06; A lower in 12% of pairs) → worse
  - M-orc vs U: mean Δ +0.364 [+0.199, +0.516] (×2.31; A lower in 12% of pairs) → worse
  - M-orc vs TUR: mean Δ +0.441 [+0.257, +0.620] (×2.76; A lower in 12% of pairs) → worse
  - S-orc vs U: mean Δ -0.122 [-0.306, +0.035] (×0.76; A lower in 71% of pairs) → no decision
- **saddle_k16 (dial 17)**
  - M-orc vs S-orc: mean Δ +0.632 [+0.470, +0.793] (×4.28; A lower in 4% of pairs) → worse
  - M-orc vs U: mean Δ +0.458 [+0.277, +0.619] (×2.87; A lower in 8% of pairs) → worse
  - M-orc vs TUR: mean Δ +0.665 [+0.517, +0.807] (×4.63; A lower in 4% of pairs) → worse
  - S-orc vs U: mean Δ -0.174 [-0.267, -0.079] (×0.67; A lower in 67% of pairs) → no decision
- **saddle_k64 (dial 65)**
  - M-orc vs S-orc: mean Δ -0.298 [-0.551, -0.017] (×0.50; A lower in 79% of pairs) → **beats**
  - M-orc vs U: mean Δ -0.760 [-0.971, -0.512] (×0.17; A lower in 92% of pairs) → **beats**
  - M-orc vs TUR: mean Δ -0.524 [-0.805, -0.200] (×0.30; A lower in 83% of pairs) → **beats**
  - S-orc vs U: mean Δ -0.462 [-0.505, -0.418] (×0.35; A lower in 100% of pairs) → **beats**

Family complete: True; M beats S at dial settings: [False, False, False, True]; supported in family: False

## davis_skodje

- **davis_skodje_g3 (dial 2)**
  - M-orc vs S-orc: mean Δ +0.088 [+0.010, +0.169] (×1.23; A lower in 25% of pairs) → worse
  - M-orc vs U: mean Δ +0.011 [-0.071, +0.125] (×1.03; A lower in 50% of pairs) → no decision
  - M-orc vs TUR: mean Δ -0.081 [-0.186, +0.026] (×0.83; A lower in 58% of pairs) → no decision
  - S-orc vs U: mean Δ -0.077 [-0.171, +0.029] (×0.84; A lower in 58% of pairs) → no decision
  - M-self vs M-orc: mean Δ +0.038 [-0.021, +0.095] (×1.09; A lower in 50% of pairs) → no decision
  - M-self vs S-self: mean Δ +0.148 [+0.065, +0.232] (×1.40; A lower in 25% of pairs) → worse
- **davis_skodje_g6 (dial 5)**
  - M-orc vs S-orc: mean Δ +0.391 [+0.321, +0.450] (×2.46; A lower in 8% of pairs) → worse
  - M-orc vs U: mean Δ +0.303 [+0.170, +0.441] (×2.01; A lower in 12% of pairs) → worse
  - M-orc vs TUR: mean Δ -0.018 [-0.115, +0.086] (×0.96; A lower in 50% of pairs) → no decision
  - S-orc vs U: mean Δ -0.088 [-0.192, +0.012] (×0.82; A lower in 58% of pairs) → no decision
  - M-self vs M-orc: mean Δ -0.060 [-0.165, +0.037] (×0.87; A lower in 67% of pairs) → no decision
  - M-self vs S-self: mean Δ +0.308 [+0.195, +0.425] (×2.03; A lower in 17% of pairs) → worse
- **davis_skodje_g18 (dial 17)**
  - M-orc vs S-orc: mean Δ +0.250 [+0.122, +0.403] (×1.78; A lower in 21% of pairs) → worse
  - M-orc vs U: mean Δ +0.151 [+0.031, +0.280] (×1.42; A lower in 25% of pairs) → worse
  - M-orc vs TUR: mean Δ -0.446 [-0.530, -0.351] (×0.36; A lower in 96% of pairs) → **beats**
  - S-orc vs U: mean Δ -0.099 [-0.165, -0.033] (×0.80; A lower in 67% of pairs) → no decision
  - M-self vs M-orc: mean Δ +0.014 [-0.023, +0.057] (×1.03; A lower in 46% of pairs) → no decision
  - M-self vs S-self: mean Δ +0.286 [+0.161, +0.424] (×1.93; A lower in 8% of pairs) → worse
- **davis_skodje_g66 (dial 65)**
  - M-orc vs S-orc: mean Δ +0.013 [-0.079, +0.087] (×1.03; A lower in 29% of pairs) → no decision
  - M-orc vs U: mean Δ -0.056 [-0.131, +0.000] (×0.88; A lower in 71% of pairs) → no decision
  - M-orc vs TUR: mean Δ -0.376 [-0.482, -0.287] (×0.42; A lower in 100% of pairs) → **beats**
  - S-orc vs U: mean Δ -0.069 [-0.090, -0.049] (×0.85; A lower in 88% of pairs) → no decision
  - M-self vs M-orc: mean Δ +0.003 [-0.001, +0.008] (×1.01; A lower in 38% of pairs) → no decision
  - M-self vs S-self: mean Δ +0.017 [-0.064, +0.091] (×1.04; A lower in 25% of pairs) → no decision

Family complete: True; M beats S at dial settings: [False, False, False, False]; supported in family: False

**H1: partial**

## Descriptive table

| system | T | method | n | median log10 E_rms [IQR] | median log10 E_max | E_max<1 |
|---|---|---|---:|---|---:|---:|
| davis_skodje_g18 | T2 | M-orc | 24 | -3.36 [-3.42, -3.17] | -2.48 | 24/24 |
| davis_skodje_g18 | T2 | M-self | 24 | -3.36 [-3.40, -3.14] | -2.46 | 24/24 |
| davis_skodje_g18 | T2 | S-orc | 24 | -3.57 [-3.78, -3.38] | -2.97 | 24/24 |
| davis_skodje_g18 | T2 | S-self | 24 | -3.59 [-3.75, -3.42] | -2.99 | 24/24 |
| davis_skodje_g18 | T2 | TUR | 24 | -2.83 [-2.93, -2.77] | -2.08 | 24/24 |
| davis_skodje_g18 | T2 | U | 24 | -3.50 [-3.61, -3.30] | -2.85 | 24/24 |
| davis_skodje_g3 | T2 | M-orc | 24 | -3.98 [-4.21, -3.65] | -3.81 | 24/24 |
| davis_skodje_g3 | T2 | M-self | 24 | -3.95 [-4.14, -3.67] | -3.71 | 24/24 |
| davis_skodje_g3 | T2 | S-orc | 24 | -4.04 [-4.25, -3.91] | -3.83 | 24/24 |
| davis_skodje_g3 | T2 | S-self | 24 | -4.03 [-4.25, -3.93] | -3.82 | 24/24 |
| davis_skodje_g3 | T2 | TUR | 24 | -3.92 [-4.09, -3.61] | -3.63 | 24/24 |
| davis_skodje_g3 | T2 | U | 24 | -3.99 [-4.19, -3.79] | -3.78 | 24/24 |
| davis_skodje_g6 | T2 | M-orc | 24 | -3.76 [-3.97, -3.57] | -3.43 | 24/24 |
| davis_skodje_g6 | T2 | M-self | 24 | -3.81 [-4.11, -3.64] | -3.48 | 24/24 |
| davis_skodje_g6 | T2 | S-orc | 24 | -4.25 [-4.32, -4.05] | -3.83 | 24/24 |
| davis_skodje_g6 | T2 | S-self | 24 | -4.17 [-4.35, -4.01] | -3.79 | 24/24 |
| davis_skodje_g6 | T2 | TUR | 24 | -3.86 [-4.01, -3.56] | -3.38 | 24/24 |
| davis_skodje_g6 | T2 | U | 24 | -4.11 [-4.36, -3.86] | -3.84 | 24/24 |
| davis_skodje_g66 | T2 | M-orc | 24 | -2.24 [-2.38, -2.11] | -1.28 | 24/24 |
| davis_skodje_g66 | T2 | M-self | 24 | -2.23 [-2.38, -2.12] | -1.28 | 24/24 |
| davis_skodje_g66 | T2 | S-orc | 24 | -2.22 [-2.39, -2.10] | -1.39 | 24/24 |
| davis_skodje_g66 | T2 | S-self | 24 | -2.23 [-2.40, -2.08] | -1.39 | 24/24 |
| davis_skodje_g66 | T2 | TUR | 24 | -1.85 [-1.90, -1.79] | -1.13 | 24/24 |
| davis_skodje_g66 | T2 | U | 24 | -2.16 [-2.29, -2.04] | -1.33 | 24/24 |
| saddle_k1 | T2 | M-orc | 24 | -3.63 [-3.83, -3.54] | -3.35 | 24/24 |
| saddle_k1 | T2 | S-orc | 24 | -3.98 [-4.06, -3.88] | -3.60 | 24/24 |
| saddle_k1 | T2 | TUR | 24 | -3.94 [-4.11, -3.70] | -3.65 | 24/24 |
| saddle_k1 | T2 | U | 24 | -3.73 [-3.92, -3.58] | -3.52 | 24/24 |
| saddle_k16 | T2 | M-orc | 24 | -2.51 [-2.61, -2.36] | -1.69 | 24/24 |
| saddle_k16 | T2 | S-orc | 24 | -3.17 [-3.26, -2.90] | -2.49 | 24/24 |
| saddle_k16 | T2 | TUR | 24 | -3.14 [-3.34, -2.95] | -2.79 | 24/24 |
| saddle_k16 | T2 | U | 24 | -2.94 [-3.16, -2.77] | -2.62 | 24/24 |
| saddle_k4 | T2 | M-orc | 24 | -3.08 [-3.29, -2.85] | -2.62 | 24/24 |
| saddle_k4 | T2 | S-orc | 24 | -3.58 [-3.69, -3.40] | -3.01 | 24/24 |
| saddle_k4 | T2 | TUR | 24 | -3.47 [-3.70, -3.31] | -3.21 | 24/24 |
| saddle_k4 | T2 | U | 24 | -3.49 [-3.69, -3.21] | -3.18 | 24/24 |
| saddle_k64 | T2 | M-orc | 24 | -1.51 [-1.62, -1.34] | -0.64 | 24/24 |
| saddle_k64 | T2 | S-orc | 24 | -1.13 [-1.48, -0.84] | -0.71 | 24/24 |
| saddle_k64 | T2 | TUR | 24 | -0.79 [-1.23, -0.60] | -0.52 | 21/24 |
| saddle_k64 | T2 | U | 24 | -0.69 [-0.93, -0.39] | -0.37 | 20/24 |
