# Turinici-style comparison on the Gate 3 systems (saved runs, no new training)

Median over 36 networks. Success = error below 1 over the whole window.

## Rössler, window [0, 10]

| method | success | median final-time error | × vs uniform | median RMS error | × vs uniform | GPU s (36 nets) |
|---|---|---|---|---|---|---|
| uniform | 4/36 | 3.19 | 1 | 1.97 | 1 | 225 |
| Turinici | 1/36 | 5.61 | 1.8 | 3.58 | 1.8 | 243 |
| scalar weight (no direction) | 13/36 | 5.6 | 1.8 | 2.27 | 1.2 | 821 |
| matrix (Lyapunov directions) | 30/36 | 0.0422 | 0.013 | 0.0314 | 0.016 | 736 |

Figure: `results/figures/turinici_style_rossler_T10.png`

## Rössler, window [0, 5]

| method | success | median final-time error | × vs uniform | median RMS error | × vs uniform | GPU s (36 nets) |
|---|---|---|---|---|---|---|
| uniform | 18/36 | 0.216 | 1 | 0.164 | 1 | 228 |
| Turinici | 19/36 | 0.24 | 1.1 | 0.171 | 1 | 231 |
| scalar weight (no direction) | 21/36 | 0.0226 | 0.1 | 0.00941 | 0.057 | 846 |
| matrix (Lyapunov directions) | 25/36 | 0.00357 | 0.016 | 0.00413 | 0.025 | 737 |

Figure: `results/figures/turinici_style_rossler_T5.png`

## Lorenz-96 (N=8), window [0, 1]

| method | success | median final-time error | × vs uniform | median RMS error | × vs uniform | GPU s (36 nets) |
|---|---|---|---|---|---|---|
| uniform | 29/36 | 0.187 | 1 | 0.107 | 1 | 251 |
| Turinici | 13/36 | 2.44 | 13 | 1.04 | 9.7 | 210 |
| scalar weight (no direction) | 36/36 | 0.057 | 0.3 | 0.0234 | 0.22 | 1118 |
| matrix (Lyapunov directions) | 36/36 | 0.0151 | 0.081 | 0.0111 | 0.1 | 752 |

Figure: `results/figures/turinici_style_lorenz96_n8_T1.png`

## Lorenz-96 (N=8), window [0, 2]

| method | success | median final-time error | × vs uniform | median RMS error | × vs uniform | GPU s (36 nets) |
|---|---|---|---|---|---|---|
| uniform | 0/36 | 14.9 | 1 | 12.2 | 1 | 187 |
| Turinici | 0/36 | 15 | 1 | 13.1 | 1.1 | 265 |
| scalar weight (no direction) | 1/36 | 12.2 | 0.81 | 7.86 | 0.64 | 1102 |
| matrix (Lyapunov directions) | 10/36 | 8.45 | 0.57 | 4.28 | 0.35 | 742 |

Figure: `results/figures/turinici_style_lorenz96_n8_T2.png`

