# H024: TurnOfMonth — OOS Walk-Forward Results

## Decision: KILL

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.41 |
| 95% CI | [0.02, 0.82] |
| CAGR | 3.72% |
| Max Drawdown | -34.41% |
| N Trades | 576 |
| SPY Benchmark Sharpe | 0.51 |
| Correlation with H001 | +0.363 |

## Kill reason

OOS Sharpe 0.41 < 0.51 (benchmark threshold). Does not beat buy-and-hold on a risk-adjusted basis.

## Folds (train=5y, test=2y, 12 folds)

| Fold | Sharpe |
|------|--------|
| 0 | 0.20 |
| 1 | 1.08 |
| 2 | 0.35 |
| 3 | 1.44 |
| 4 | 0.47 |
| 5 | -0.41 |
| 6 | -0.15 |
| 7 | 0.96 |
| 8 | -0.29 |
| 9 | 1.05 |
| 10 | 0.68 |
| 11 | 1.18 |

## Sensitivity (n_days)

| n_days | Sharpe |
|--------|--------|
| 2 | 0.32 |
| 3 (canonical) | 0.41 |
| 5 | 0.54 |

## Notes

- n_days=5 (5 days each end, ~10 active days/month) achieves Sharpe 0.54 — beats benchmark
- Canonical n_days=3 falls short; the TOM window needs to be wider on SPY data
- High turnover (576 trades) — switches on/off every few days; costs erode the edge
- H001 correlation 0.363 — the lower end of Batch 4; structural partial diversification
- Three folds with negative Sharpe (5, 6, 8) — signal is inconsistent across regimes
- Would consider revisiting as n_days=5 if running a broader parameter sweep
</content>
