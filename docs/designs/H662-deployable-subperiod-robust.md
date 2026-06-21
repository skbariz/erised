# H662 — DeployableSubperiodRobust662

**Signal:** Deployable counterpart of H650, but the pool weights are fit to maximize the **minimum Sharpe
across the two equal calendar halves** of the OOS period — max min(Sharpe_firsthalf, Sharpe_secondhalf) — on
the **1-day-lagged** VIX-momentum regime, over the interior-only grid (threshold {15,16,17} × window {20},
plain + asymmetric). Full battery: 5-start weight stability, threshold sensitivity, gap vs H650.

**Why it should work:** a pooled-Sharpe optimum can lean on one macro era (e.g., the post-2009 QE bull) and
quietly underperform in the other. Maximizing the *worse* half is a minimax/regime-robust criterion: it
forces the allocation to earn its Sharpe in both the crisis-heavy first half (which contains 2000–02 and
2008) and the calmer second half, rather than averaging a great era against a mediocre one.

**Expected behavior:** the fit should up-weight sleeves that work in both eras (parity, seasonal, L/S
reversal) and trim sleeves whose edge is concentrated in one macro regime; realized half-Sharpes should be
more balanced than H650's, possibly at a small cost to the pooled Sharpe.

**Success:** an interior, lag-fitted config clearing the keep bar with a higher *worse-half* Sharpe than
H650's worse half, weight-stable. **Falsify:** balancing the halves drops the strategy below the keep bar,
or the minimax fit coincides with H650 → no era concentration to correct.

**Result:** appended after run.
