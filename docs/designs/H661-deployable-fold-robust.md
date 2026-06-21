# H661 — DeployableFoldRobust661

**Signal:** Deployable counterpart of H650, but the pool weights are fit to maximize
**mean(per-fold OOS Sharpe) − λ·std(per-fold Sharpe)** (λ = 0.5) on the **1-day-lagged** VIX-momentum
regime, over the interior-only grid (threshold {15,16,17} × window {20}, plain + asymmetric). Full battery:
5-start weight stability, threshold sensitivity, gap vs H650.

**Why it should work:** the program treats per-fold Sharpe *dispersion* as the primary overfitting signal.
Every other objective optimizes a pooled, full-sample statistic and only *reports* fold variance after the
fact. This objective optimizes it directly — penalizing weights whose edge is concentrated in a few lucky
folds — so the fitted allocation should be the one whose OOS Sharpe is most uniform across the 12 folds.

**Expected behavior:** vs the pooled-Sharpe H650 (fold range 1.21–2.75), the fold-robust fit should compress
the fold range (raise the worst fold, perhaps lower the best) at a small cost to the mean — trading peak
pooled Sharpe for consistency.

**Success:** an interior, lag-fitted config clearing the keep bar with a *tighter* min..max fold-Sharpe
spread than H650 at comparable mean, weight-stable. **Falsify:** the fold-robust fit cannot tighten the
spread without dropping below the keep bar, or it collapses to H650 (λ inactive) → fold-variance targeting
adds nothing.

**Result:** appended after run.
