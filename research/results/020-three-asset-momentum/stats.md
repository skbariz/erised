# H020: ThreeAssetMomentum — OOS Walk-Forward Results

## Decision: KILL

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.47 |
| 95% CI | [0.06, 0.86] |
| CAGR | 6.04% |
| Max Drawdown | -31.29% |
| N Trades | 95 |
| SPY Benchmark Sharpe | 0.51 |
| Correlation with H001 | +0.329 |

## Kill reason

OOS Sharpe 0.47 < 0.51 (benchmark threshold). Does not beat buy-and-hold on a risk-adjusted basis.

## Folds (train=5y, test=2y, 12 folds)

| Fold | Sharpe |
|------|--------|
| 0 | 0.00 (no TLT/GLD data — cash) |
| 1 | 0.00 (no TLT/GLD data — cash) |
| 2 | 0.00 (GLD data begins 2004 — first fold with GLD) |
| 3 | 0.43 |
| 4 | 1.15 |
| 5 | -0.05 |
| 6 | 1.18 |
| 7 | 1.37 |
| 8 | -0.25 |
| 9 | 1.23 |
| 10 | 0.41 |
| 11 | 0.15 |

## Sensitivity (lookback_months)

| lookback | Sharpe |
|----------|--------|
| 6m | 0.30 |
| 9m | 0.23 |
| 12m (canonical) | 0.47 |

## Notes

- Correlation with H001 is 0.329 — closest of Batch 3 to the 0.30 ensemble threshold
- First 3 folds produce zero returns because TLT/GLD data doesn't exist for those test windows
- This data-scarcity penalty drags the aggregate Sharpe from ~0.65 (folds 3-11) to 0.47
- Sensitivity shows degradation at shorter lookbacks (0.30/0.23); not robust
- Would be reconsidered if restated over the 2004+ period (TLT+GLD available)
