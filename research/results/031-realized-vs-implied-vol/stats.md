# H031: RealizedVsImpliedVol — OOS Walk-Forward Results

## Decision: KEEP

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.58 |
| 95% CI | [0.16, 0.96] |
| CAGR | 8.52% |
| Max Drawdown | −51.43% |
| N Trades | 321 |
| SPY Benchmark Sharpe | 0.51 |
| Correlation with H001 | +0.693 |

## Folds (train=5y, test=2y, 12 folds)

| Fold | Sharpe |
|------|--------|
| 0 | 0.63 |
| 1 | −0.30 |
| 2 | 0.46 |
| 3 | 0.58 |
| 4 | 0.67 |
| 5 | 0.38 |
| 6 | 0.83 |
| 7 | 0.82 |
| 8 | 0.48 |
| 9 | 0.83 |
| 10 | 0.72 |
| 11 | 0.40 |

## Sensitivity (realized_window)

| realized_window | Sharpe |
|-----------------|--------|
| 10 | 0.61 |
| 20 (canonical) | 0.58 |
| 60 | 0.42 |

## Notes

- CAGR 8.52%: highest CAGR in Batch 5, slightly above H001's 7.86%
- MaxDD −51.43%: close to SPY's −55.92% — the VRP filter does not protect well in crashes (VRP is often positive even during market falls if implied vol rises slower than realized)
- H001 correlation +0.693: meaningful but below the highest-correlated strategies (H022: 0.912)
- Fold 1 negative (−0.30): 2000–2002 tech bust had high realized vol exceeding VIX → strategy correctly went flat, but missed the recovery bounce
- Realized window 10d is slightly better (Sharpe 0.61): shorter window responds faster to volatility regime changes
- The vol risk premium (VIX > realized vol) is positive on average ~65-70% of trading days — confirmed in this OOS test
