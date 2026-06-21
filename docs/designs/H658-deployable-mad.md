# H658 — DeployableMAD658

**Signal:** Deployable counterpart of H650, but the pool weights are fit to maximize the **leveraged
return-to-MAD ratio** (annualized mean ÷ mean-absolute-deviation) on the **1-day-lagged** VIX-momentum
regime, over the interior-only grid (threshold {15,16,17} × window {20}, plain + asymmetric). Full battery:
5-start weight stability, threshold sensitivity, gap vs H650.

**Why it should work:** MAD is an L1 dispersion measure — it weights each deviation linearly, where Sharpe's
variance (L2) squares them and so is dominated by a handful of extreme days. On a pool with occasional
outlier days, MAD-fitting is more robust to those outliers and may produce a steadier deployable allocation
that the variance objective over-fits to the tails of the in-sample window.

**Expected behavior:** for near-Gaussian sleeves MAD ≈ 0.8·σ, so the ratio ≈ 1.25·Sharpe and weights should
track H650 closely; divergence should appear only where a sleeve has fat-but-symmetric tails that L2 over-
penalizes.

**Success:** an interior, lag-fitted config with deployable Sharpe ≥ 1.7 and a MAD-ratio at least matching
the Sharpe-optimal H650's, weight-stable. **Falsify:** MAD-fitting underperforms H650 on both Sharpe and the
MAD ratio, or no interior config clears Sharpe 1.7 → L1 dispersion adds nothing on this pool.

**Result:** appended after run.
