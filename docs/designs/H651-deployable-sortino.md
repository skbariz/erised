# H651 — DeployableSortino651

**Signal:** Deployable counterpart of the H650 terminal champion, but the pool weights are fit to
maximize the **leveraged Sortino ratio** (downside-deviation denominator) on the **1-day-lagged**
VIX-momentum regime, over the same interior-only grid (threshold {15,16,17} × window {20}, plain +
asymmetric caps). Full battery: 5-start weight stability, threshold sensitivity, gap vs H650.

**Why it should work:** Sharpe penalizes upside and downside symmetrically; a real mandate cares about
downside. Optimizing the lagged, tradeable signal for Sortino should concentrate weight on sleeves that
protect the left tail, plausibly improving realized MaxDD / Calmar at little Sharpe cost.

**Success:** an interior, lag-fitted config with deployable Sharpe ≥ 1.7 and Sortino ≥ H650's, weight-
stable across restarts. **Falsify:** Sortino-fitting underperforms H650 on both Sharpe and Sortino, or no
interior config clears Sharpe 1.7 → downside-targeting adds nothing here.

**Result (KEEP).** Config t15w20 (interior). Deployable **Sharpe 1.969** [1.57, 2.35], **Sortino 2.067**,
Calmar 1.276, CAGR 9.68%, MaxDD −7.59%, H001 corr +0.463, folds 1.21–2.79. 5-start Sortino range 0.0023
(weight-stable). Sortino-fitting beats the Sharpe-optimal H650 on Sortino (2.067 vs 2.049) at a negligible
Sharpe cost (−0.011) — downside-targeting delivers marginally better tail efficiency, as hypothesized.
