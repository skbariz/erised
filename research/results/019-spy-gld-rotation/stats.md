# H019: SPYGLDRotation — OOS Walk-Forward Results

## Decision: KEEP (high H001 correlation limits ensemble value)

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.63 |
| 95% CI | [0.22, 1.03] |
| CAGR | 9.68% |
| Max Drawdown | -33.72% |
| N Trades | 97 |
| SPY Benchmark Sharpe | 0.51 |
| Correlation with H001 | +0.788 |

## Folds (train=5y, test=2y, 12 folds)

| Fold | Sharpe |
|------|--------|
| 0 | 0.93 |
| 1 | 0.04 |
| 2 | 1.05 |
| 3 | 0.77 |
| 4 | 0.33 |
| 5 | 0.46 |
| 6 | 0.73 |
| 7 | 1.44 |
| 8 | 0.28 |
| 9 | 1.95 |
| 10 | 0.25 |
| 11 | 0.53 |

## Sensitivity (lookback_months)

| lookback | Sharpe |
|----------|--------|
| 6m | 0.62 |
| 9m | 0.60 |
| 12m (canonical) | 0.63 |

## Notes

- Always-invested structure (no cash) gives higher CAGR (9.68%) than H018 (7.34%)
- MaxDD still -33.72% — GLD hedge doesn't reduce peak drawdown vs SPY-only strategies
- Correlation with H001 is 0.788 — the highest of Batch 3; primarily follows SPY trend
- GLD's value is in regime diversification, but 12m SPY momentum already trends with equity
- Lookback sensitivity is tight (0.60-0.63); signal is robust to parameter choice
