# PORTAL vs BBOB — ELA Feature Coverage

48 PORTAL instances · 72 BBOB (f1–f24, i1–3) · dim=2 · LHS 500 pts

---

## Feature ranges

| Feature | BBOB | PORTAL | Coverage |
|---|---|---|---|
| `ela_distr.skewness` | [−0.04, 4.78] | [−0.09, 2.89] | 62% |
| `fitness_distance.fitness_std` | [0.00, 0.05] | [0.00, 0.07] | **152%** |
| `nbc.nn_nb.sd_ratio` | [0.19, 0.73] | [0.25, 0.72] | 87% |
| `nbc.nb_fitness.cor` | [−0.68, −0.08] | [−0.38, −0.10] | 46% |
| `ela_meta.lin_simple.adj_r2` | [−0.00, 1.00] | [0.01, 0.92] | 91% |
| `ela_meta.lin_w_interact.adj_r2` | [−0.01, 1.00] | [0.05, 0.92] | 87% |
| `ela_meta.quad_simple.adj_r2` | [−0.01, 1.00] | [0.12, 1.00] | 87% |
| `ela_meta.quad_w_interact.adj_r2` | [−0.02, 1.00] | [0.36, 1.00] | 63% |

---

## Why two gaps cannot be closed

### `nbc.nb_fitness.cor` (46%) and `ela_distr.skewness` high tail (>2.9)

Both extremes are owned exclusively by **BBOB f23 Katsuura**, which uses a deterministic quasi-random frequency sum:

$$f(x) = -\frac{10}{d^2} \sum_{i=1}^{d} \sum_{k=0}^{31} \frac{|2^k x_i - \text{round}(2^k x_i)|}{2^k}$$

This produces a *fractal, space-filling* ruggedness that drives fitness–neighbour correlation to −0.68 and skewness to 4.78.
PORTAL's transform layer (additive-periodic, log-sinusoidal, wavelet) generates *smooth oscillations with a single spatial scale*; it cannot replicate multi-octave fractal noise.

### `ela_meta.quad_w_interact.adj_r2` low tail (<0.36)

The only BBOB instances near 0 are **f16 Weierstrass** (quasi-fractal sum of cosines) and **f23 Katsuura**.
Same root cause: a quadratic model explains almost nothing when the landscape has no dominant curvature at any scale.
PORTAL always has a well-defined basin (Form A or B), so the quadratic component explains ≥ 36% of variance in every instance.

---

## Most unique PORTAL instances

Ranked by L2 distance to nearest BBOB neighbour in jointly-normalised 8D feature space:

| Rank | Instance | Dist | What makes it unique |
|---|---|---|---|
| 1 | `P41_I1_LogSin_SubQuad` | 0.781 | log-sinusoidal + sub-quadratic curvature (p=0.3) — flat rugged basin absent in BBOB |
| 2 | `P13_B1_SubQuadratic` | 0.543 | pure sub-quadratic bowl (p=0.3) — BBOB has no flat-sided basins |
| 3 | `P37_H1_K12_Smooth` | 0.540 | 12 smooth basins — BBOB tops out at ~20 local optima but with structure |
| 4 | `P20_C4_RippleMultimodal` | 0.536 | 3 basins each with additive-periodic ripples |
| 5 | `P45_J1_K10_IllCond` | 0.492 | 10 basins with mild conditioning |
| 6 | `P3_IllConditionedCuspRidge` | 0.486 | radial + log-sin + tensor in one landscape |
| 7 | `P44_I4_Additive_SubQuad` | 0.478 | additive-periodic + sub-quadratic (different from #1: no conditioning) |
| 8 | `P22_D2_StrongLogSin_Rot` | 0.478 | rotated log-sinusoidal, balanced ill-conditioning |
