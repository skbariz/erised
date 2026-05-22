# H018: SPYTLTDualMomentum — OOS Walk-Forward Results

## Decision: KEEP (high H001 correlation limits ensemble value)

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.57 |
| 95% CI | [0.17, 0.98] |
| CAGR | 7.34% |
| Max Drawdown | -33.72% |
| N Trades | 79 |
| SPY Benchmark Sharpe | 0.51 |
| Correlation with H001 | +0.687 |

## Folds (train=5y, test=2y, 12 folds)

| Fold | Sharpe |
|------|--------|
| 0 | 0.00 (no TLT data — in cash) |
| 1 | 0.00 (no TLT data — in cash) |
| 2 | 1.40 |
| 3 | 0.77 |
| 4 | 0.33 |
| 5 | 0.51 |
| 6 | 0.46 |
| 7 | 1.49 |
| 8 | 0.31 |
| 9 | 2.13 |
| 10 | 0.21 |
| 11 | 0.50 |

## Sensitivity (lookback_months)

| lookback | Sharpe |
|----------|--------|
| 6m | 0.68 |
| 9m | 0.53 |
| 12m (canonical) | 0.57 |

## Notes

- Beats benchmark (Sharpe 0.57 vs 0.51), CI lower bound > 0 → keeps per kill criteria
- MaxDD identical to SPY benchmark (-33.72%); no meaningful downside protection
- Correlation with H001 is 0.687 — too high for ensemble diversification value
- Folds 0-1 show 0.00 because TLT data begins 2002; strategies in cash during 1993-2001 test windows
- 6m lookback slightly better (0.68); canonical 12m chosen for robustness
