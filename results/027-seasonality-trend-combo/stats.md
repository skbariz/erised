# H027: SeasonalityTrendCombo — OOS Walk-Forward Results

## Decision: KEEP (high H001 correlation — minimal ensemble value)

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.61 |
| 95% CI | [0.19, 1.03] |
| CAGR | 5.70% |
| Max Drawdown | -33.72% |
| N Trades | 49 |
| SPY Benchmark Sharpe | 0.51 |
| Correlation with H001 | +0.712 |

## Folds (train=5y, test=2y, 12 folds)

| Fold | Sharpe |
|------|--------|
| 0 | 1.20 |
| 1 | 0.34 |
| 2 | 0.62 |
| 3 | 0.63 |
| 4 | 0.56 |
| 5 | 0.61 |
| 6 | 1.66 |
| 7 | 1.31 |
| 8 | 0.21 |
| 9 | 2.45 |
| 10 | 0.20 |
| 11 | 0.01 |

## Sensitivity (MA params)

| MA params | Sharpe |
|-----------|--------|
| 20/100 | 0.57 |
| 50/200 (canonical) | 0.61 |
| 50/150 | 0.59 |

## Notes

- Matches H001's OOS Sharpe (0.61) while being active only Nov-Apr
- Lower CAGR (5.70% vs H001's 7.86%) because May-Oct returns are abandoned
- MaxDD identical to H001 (-33.72%) — drawdowns occur in favorable months too
- H001 correlation 0.712 — lower than pure H001 since ~50% of H001's return
  is excluded, but still too high for ensemble use (threshold: 0.30)
- Very consistent across MA params (0.57-0.61) — the seasonal mask dominates
- Per-fold distribution is more consistent than H023 alone (minimum 0.01 vs -0.01)
