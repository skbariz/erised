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

**Result (KEEP).** Config t15w20-asym. Sharpe **1.953** [1.56, 2.34], return-to-MAD ratio 2.684, Calmar
**1.427** (Set14 high), Sortino 2.035, CAGR 9.56%, MaxDD **−6.70%** (Set14 low), H001 corr +0.439, folds
1.30–2.73. The L1-dispersion objective is a near-twin of the Omega fit (H657) and posts the lowest MaxDD and
highest Calmar of the whole set — linear (vs squared) penalty on deviations yields the steadiest deployable
book, confirming that L2 variance over-weights a handful of outlier days that L1 ignores.
