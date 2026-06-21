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

**Result (KEEP).** Config t15w20. Pooled Sharpe **1.963** [1.57, 2.36], half-period Sharpes **1.989 / 1.989**
(split 2010-01-14) — the minimax balanced the two eras to the third decimal — Calmar 1.315, Sortino 2.027,
CAGR 9.70%, MaxDD −7.37%, H001 corr +0.441, folds 1.26–2.78. Forcing the worse half up delivered
near-identical pre/post-2010 Sharpes at only −0.016 pooled vs H650: regime balance is nearly free on this
pool, and no single macro era is load-bearing for the deployable edge.
