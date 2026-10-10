# H655 — DeployableCagrDDfloor655

**Signal:** A growth-with-floor deployable mandate: fit pool weights on the 1-day-lagged signal to maximize
**leveraged CAGR subject to a −10% MaxDD floor** (quadratic penalty on drawdown beyond the floor), over the
interior grid (threshold {15,16,17} × window {20}, plain + asymmetric). Deployable counterpart of the
same-day H635 / H566 CAGR-with-floor work.

**Why it should work:** Sharpe-optimal weights leave CAGR on the table; a CAGR objective with an explicit
drawdown floor targets the growth a leveraged mandate actually wants while capping tail risk — a different
point on the deployable frontier than the Sharpe / Sortino / Calmar champions.

**Success:** an interior config with higher deployable CAGR than H650 (10.08%) while holding MaxDD near the
−10% floor, weight-stable. **Falsify:** the floor cannot be held without collapsing Sharpe below the keep
bar, or CAGR fails to beat H650 → the growth and floor objectives conflict here.

**Result (KEEP).** Config t17w20 (interior). Deployable **CAGR 12.22%** (best on the deployable frontier,
+2.14% vs H650's 10.08%), Sharpe 1.751 [1.34, 2.14], Calmar 1.209, Sortino 1.777, MaxDD **−10.11%**
(essentially at the −10% floor), H001 corr +0.424, folds 0.75–3.18. The growth objective trades ~0.23
Sharpe for a materially higher CAGR while holding the drawdown floor to within 0.11pp — a distinct
growth-mandate point on the deployable frontier.
