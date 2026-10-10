# H653 — DeployableInverseVol653

**Signal:** A **non-optimized** deployable baseline: pool weights set by **inverse realized volatility**
(risk-parity weighting, using each sleeve's available-history vol), with the same 1-day-lagged interior-grid
leverage as H650. No SLSQP — the weights cannot be overfit. Sweep the interior grid (threshold {15,16,17} ×
window {20}, plain + asymmetric) and finalize on the best config; sensitivity over threshold.

**Why it should work:** inverse-vol is the canonical un-overfittable heuristic. As a deployable baseline it
isolates how much of H650's Sharpe comes from the leverage mechanism vs the optimized weights — and whether
a principled heuristic beats the equal-weight baseline (H648: 1.383).

**Success:** deployable Sharpe between equal-weight (H648 1.383) and optimized (H650 1.980), confirming the
weighting-scheme ladder. **Falsify:** inverse-vol underperforms equal-weight → naive vol-weighting hurts
here (likely from over-weighting low-vol/sparse sleeves).

**Result (KEEP — falsification partly realized).** Config t15w20-asym. Deployable **Sharpe 1.303**
[0.91, 1.69], Calmar 0.868, Sortino 1.392, CAGR 7.00%, MaxDD −8.06%, H001 corr +0.683, folds 0.52–2.04.
Inverse-vol lands **below equal-weight (H648: 1.383, −0.080)** and far below optimized (H650: 1.980,
−0.677): naive vol-weighting over-weights low-vol/sparse sleeves, hurting the blend. Still clears the keep
bar, but the weighting-scheme ladder is equal-weight > inverse-vol here — optimization earns its ~0.6 Sharpe.
