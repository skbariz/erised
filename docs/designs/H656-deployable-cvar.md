# H656 — DeployableCVaR656

**Signal:** Deployable counterpart of H650, but the pool weights are fit to maximize the **leveraged
return-to-CVaR ratio** (annualized mean ÷ 5% expected shortfall) on the **1-day-lagged** VIX-momentum
regime, over the same interior-only grid (threshold {15,16,17} × window {20}, plain + asymmetric). Full
battery: 5-start weight stability, threshold sensitivity, gap vs H650.

**Why it should work:** Sharpe's variance denominator penalizes upside and downside equally and is blind to
tail shape. A leveraged mandate fears the left tail specifically. CVaR (the mean of the worst 5% of days)
targets that tail directly, plausibly concentrating weight on sleeves that truncate crash days — a distinct
point on the deployable frontier from the Sharpe / Sortino / CAGR champions.

**Expected behavior:** in 2008/2020 the CVaR objective should down-weight high-beta sleeves whose worst days
cluster, trading a little pooled Sharpe for a better tail ratio. In calm 2013–2017 it should look much like
H650 (the tail is thin, so CVaR ≈ scaled vol).

**Success:** an interior, lag-fitted config with deployable Sharpe ≥ 1.7 and a higher return-to-CVaR ratio
than the Sharpe-optimal H650, weight-stable across restarts. **Falsify:** CVaR-fitting underperforms H650
on *both* Sharpe and the tail ratio, or no interior config clears Sharpe 1.7 → tail-targeting adds nothing
on this pool.

**Result:** appended after run.
