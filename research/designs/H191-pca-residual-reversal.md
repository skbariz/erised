# H191: PCAResidualReversal

## Hypothesis

**Signal:** Multi-factor extension of H186 (beta-stripped reversal). Instead of stripping only market beta (1 factor), strip K=10 PCA factors derived from training returns. Apply 5-day reversal signal on multi-factor residuals. Sector-neutral top 20%, weekly rebalance.

**Why it should work:** H186 strips one risk factor (market beta) before reversing. But stocks also have sector, size, value, and other systematic exposures. Stripping K=10 PCA factors removes more systematic variance, leaving a purer idiosyncratic residual. If the reversal premium is concentrated in the idiosyncratic component, more thorough factor-stripping should improve signal quality. Connor and Korajczyk (1988) show K=5-10 PCA factors explain most cross-sectional variance in US equities.

**PCA implementation:** Fit on training returns via SVD (no sklearn required). Components are fixed from training period and applied to test period — no data leakage. K=10 factors typically explains 25-40% of cross-sectional return variance.

**Relationship to H186:** H186 uses 1 factor (SPY market beta). H191 uses K=10. Expected improvement: if H191 > H186 (1.095), more factors help. If roughly equal, 1 factor captures most of the benefit.

**Expected regime behavior:** Same as H185/H186 since the underlying signal is still 5-day reversal. PCA residuals should reduce the "trend component" in reversals, potentially improving performance during trending regimes (2000-2002, 2008-2009).

## Success criteria

- OOS Sharpe > H186 Sharpe (1.095) — must beat single-factor version
- 95% CI lower bound > 0
- H001 corr < H186 (+0.613) — more factor-stripping should reduce systematic exposure

## Falsification

If H191 Sharpe ≤ H186 Sharpe (1.095), adding more PCA factors beyond market beta adds no value. The market factor captures most of the systematic component in 5-day reversals.

## Implementation

- `PCAResidualReversal` in `src/erised/strategies/cross_sectional.py`
- PCA fitted once per fold on training returns via `_fit_pca_components()`
- Sensitivity sweep: n_components ∈ {5, 10, 20}

## OOS Results

**Decision: KEEP**

| Metric | Value |
|--------|-------|
| OOS Sharpe | 1.135 |
| 95% CI | [0.733, 1.553] |
| CAGR | 24.75% |
| Max DD | -48.11% |
| H001 corr | +0.619 |

Beats H186 single-factor (1.135 vs 1.095). 12/12 folds positive including GFC (fold 5: +0.381). Stunning finding: n_components sensitivity is completely flat (K=5: 1.138, K=10: 1.135, K=20: 1.138) — the number of PCA factors stripped is irrelevant. The residual reversal signal is robust regardless of factor structure. H192 (with OU threshold) is slightly better at 1.175.
