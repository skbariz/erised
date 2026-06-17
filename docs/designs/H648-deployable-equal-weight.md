# H648 — DeployableEqualWeight

**Signal:** Equal-weight the 55-strategy pool (no optimization) + asymmetric VIX-momentum regime with the
signal **lagged 1 day**. A fully-deployable, no-optimization baseline.

**Why it should work / what it tests:** H608 showed equal-weight + same-day regime = 1.425, far below the
optimized 2.237 — but that was the unlevered-window config. This asks the cleaner question: in the
*deployable* (lagged) setting, how much does weight optimization actually buy over a naive equal-weight book?
If the gap to H643 (deployable optimized, 1.918) is small, most of the deployable edge is the regime
mechanism, not the weights — a strong robustness statement.

**Success:** KEEP (Sharpe > 0.51, CI > 0). Report the gap vs H643. **Falsify:** equal-weight ≈ H643 →
optimization adds little once execution is honest. **Result:** appended after run.
