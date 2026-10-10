# H022: DefensiveEquityHedge — OOS Walk-Forward Results

## Decision: KEEP (very high H001 correlation — minimal ensemble value)

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.68 |
| 95% CI | [0.25, 1.07] |
| CAGR | 9.60% |
| Max Drawdown | -33.72% |
| N Trades | 25 |
| SPY Benchmark Sharpe | 0.51 |
| Correlation with H001 | +0.912 |

## Folds (train=5y, test=2y, 12 folds)

| Fold | Sharpe |
|------|--------|
| 0 | 0.74 |
| 1 | -0.13 |
| 2 | 0.87 |
| 3 | 0.83 |
| 4 | 0.61 |
| 5 | 0.26 |
| 6 | 0.71 |
| 7 | 1.51 |
| 8 | 0.36 |
| 9 | 2.49 |
| 10 | 0.78 |
| 11 | 0.58 |

## Sensitivity (fast/slow)

| MA params | Sharpe |
|-----------|--------|
| 20/100 | 0.66 |
| 50/200 (canonical) | 0.68 |
| 50/150 | 0.70 |

## Notes

- Highest Sharpe of Batch 3 for strategies using pure momentum/trend signals (0.68)
- Best CAGR in the batch: 9.60% — retaining 50% SPY in downtrends vs going to cash
- MaxDD -33.72% — no improvement over H001 because downtrend allocation still holds SPY
- Correlation with H001 is 0.912 — near-identical return profile to H001; no ensemble value
- Low turnover (25 trades) — the MA(50,200) signal is slow-moving by construction
- The strategy is H001 with TLT as partial hedge, but when the MA signals are wrong
  (2001, 2008 early), 50% SPY exposure still incurs significant drawdown
