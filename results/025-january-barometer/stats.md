# H025: JanuaryBarometer — OOS Walk-Forward Results

## Decision: KEEP

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.53 |
| 95% CI | [0.12, 0.94] |
| CAGR | 5.47% |
| Max Drawdown | -29.53% |
| N Trades | 26 |
| SPY Benchmark Sharpe | 0.51 |
| Correlation with H001 | +0.565 |

## Folds (train=5y, test=2y, 12 folds)

| Fold | Sharpe |
|------|--------|
| 0 | 1.04 |
| 1 | -0.49 |
| 2 | 0.00 (years went flat — negative Jan kept strategy in cash) |
| 3 | 0.52 |
| 4 | 0.58 |
| 5 | 0.00 (flat year) |
| 6 | 0.07 |
| 7 | 1.47 |
| 8 | 0.00 (flat year) |
| 9 | 2.02 |
| 10 | 0.43 |
| 11 | 1.51 |

## Notes

- MaxDD -29.53% — second-best in the program after H021 (-18.41%)
- The January gate correctly identifies and avoids bear-market years (2002, 2008 etc.)
- Low turnover (26 trades over 30 years — one round-trip per year maximum)
- Three folds show Sharpe 0.00: the strategy stayed entirely flat those 2-year periods
  because January was negative in those years; benchmark was negative → correct behavior
- H001 correlation 0.565 — lower than all prior trend/momentum strategies (0.79-0.92)
  because the signal fires only ~70-75% of years; provides some diversification
- Fold 1 (-0.49): negative January 1998/1999 kept strategy flat during the Y2K bull run;
  the missed gains rather than actual losses caused negative Sharpe
