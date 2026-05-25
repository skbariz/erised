# H192: PCAOUMeanReversion

## Hypothesis

**Signal:** Multi-factor extension of H189 (Avellaneda-Lee OU). Instead of stripping only market beta, strip K=10 PCA factors, then apply OU z-score entry on 5-day cumulative residuals. Enter long when z-score < -1.5.

**Why it should work:** H189 (single-factor OU) produced Sharpe 1.025. The OU formalism is most valid when the residual process is truly idiosyncratic. By stripping K=10 systematic factors instead of 1, the residual is closer to a pure OU process, making the z-score threshold more statistically meaningful. Avellaneda and Lee's original paper used PCA factors (not just market beta), so H192 is closer to the original paper's methodology.

**Relationship to H189:** H189 = single-factor OU (SPY beta). H192 = K-factor OU (PCA). If H192 > H189, multi-factor residuals produce better-calibrated OU processes.

**Key difference from H191:** H191 uses raw reversal (all residuals participate). H192 uses OU z-score threshold (-1.5) — only enters when residual is statistically extreme. This should produce fewer but higher-quality trades.

**Expected regime behavior:** Same as H189 — the OU framework is most reliable in range-bound markets and can fail during regime changes (GFC, COVID crash) when the "mean" shifts permanently.

## Success criteria

- OOS Sharpe > H189 Sharpe (1.025) — must beat single-factor OU
- 95% CI lower bound > 0
- Fewer trades than H191 (threshold filtering reduces trade count)

## Falsification

If H192 Sharpe ≤ H189 Sharpe (1.025), the additional PCA factors add noise rather than signal. The OU process calibration may be better with a single well-defined market factor than with 10 noisy PCA factors.

## Implementation

- `PCAOUMeanReversion` in `src/quant/strategies/cross_sectional.py`
- Same PCA fitting as H191; OU z-score applied to 5-day cumulative residuals
- Sensitivity sweep: entry_z ∈ {-1.0, -1.5, -2.0}

## OOS Results

*(to be filled after backtest)*
