# H650 — GrandDeployable650

**Signal:** The terminal *deployable* champion. Optimizes weights **directly on the 1-day-lagged** regime
signal (so the weights are fit to what's actually tradeable, not the same-day idealization), searching an
interior-only grid (threshold {15,16,17} × window {20}, plain + asymmetric). Full battery: 5-start weight
stability, threshold/window sensitivity, and the gap vs the same-day H645.

**Why it should work:** every prior champion optimized on the same-day signal and then suffered under lag.
Fitting the weights to the lagged signal aligns the optimization with deployment, and the interior-only
grid avoids the Set10 overfitting corner. This is the honest, executable terminal strategy of the program.

**Success:** an interior, lag-fitted config with deployable Sharpe ~1.9–2.1, weight-stable, beating the
naive "optimize-then-lag" H643. **Falsify:** no lag-fitted interior config clears Sharpe 1.7 stably →
the family is not deployable. **Result:** appended after run.
