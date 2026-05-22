# H029: VIXRegimeFilter — OOS Walk-Forward Results

## Decision: KILL

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.43 |
| 95% CI | [0.01, 0.84] |
| CAGR | 3.39% |
| Max Drawdown | −25.53% |
| N Trades | 314 |
| SPY Benchmark Sharpe | 0.51 |
| Correlation with H001 | +0.562 |

## Kill reason

OOS Sharpe 0.43 < 0.51 (benchmark threshold). CI lower bound is barely positive
(+0.01) but the Sharpe itself fails. The VIX < 20 filter is too narrow — misses
too many profitable periods when VIX is between 20 and 25.

## Folds (train=5y, test=2y, 12 folds)

| Fold | Sharpe |
|------|--------|
| 0 | 0.82 |
| 1 | 0.28 |
| 2 | 0.35 |
| 3 | 0.54 |
| 4 | 0.60 |
| 5 | −0.04 |
| 6 | 0.56 |
| 7 | 0.41 |
| 8 | 0.53 |
| 9 | 0.52 |
| 10 | 0.26 |
| 11 | 0.47 |

## Sensitivity (vix_threshold)

| vix_threshold | Sharpe |
|---------------|--------|
| 15 | 0.43 |
| 20 (canonical) | 0.43 |
| 25 | 0.32 |

## Notes

- MaxDD −25.53%: second-best among Batch 5 strategies — the VIX filter keeps the portfolio flat during most crashes
- Stable across vix_threshold 15–20; threshold 25 surprisingly worse (more false positives)
- 314 trades: VIX oscillates around 20 frequently, generating many switches
- H001 correlation +0.562: moderate, but Sharpe is insufficient for standalone use
- Killed due to Sharpe below 0.51; a higher threshold (25) with a smoothing buffer
  (e.g., 5d MA of VIX) might recover the Sharpe — flagged for ensemble investigation
