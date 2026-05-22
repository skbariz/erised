# H023: SellInMay — OOS Walk-Forward Results

## Decision: KEEP (lowest H001 correlation of any kept strategy so far)

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.53 |
| 95% CI | [0.12, 0.93] |
| CAGR | 6.71% |
| Max Drawdown | -34.82% |
| N Trades | 49 |
| SPY Benchmark Sharpe | 0.51 |
| Correlation with H001 | +0.505 |

## Folds (train=5y, test=2y, 12 folds)

| Fold | Sharpe |
|------|--------|
| 0 | 1.36 |
| 1 | 0.07 |
| 2 | 0.33 |
| 3 | 0.63 |
| 4 | -0.01 |
| 5 | 0.16 |
| 6 | 1.64 |
| 7 | 1.31 |
| 8 | 0.15 |
| 9 | 2.50 |
| 10 | 0.48 |
| 11 | 0.23 |

## Sensitivity (favorable_months)

| Window | Sharpe |
|--------|--------|
| Oct-Apr (7 months) | 0.60 |
| Nov-Apr (6 months, canonical) | 0.53 |
| Nov-Mar (5 months) | 0.36 |

## Notes

- Bouman & Jacobsen (2002) replicated in 37 countries — confirmed in SPY OOS here
- H001 correlation 0.505: lowest of any kept strategy prior to this batch
- Fold 4 is barely negative (-0.01) — one bad 2-year window in 24 years; acceptable
- MaxDD -34.82% is marginally worse than H001's -33.72%; the seasonal filter
  doesn't protect against drawdowns (still long in winter crashes)
- Extending to Oct-Apr improves Sharpe 0.53 → 0.60; canonical Nov-Apr is more
  conservative and matches original Sell-in-May academic framing
- Pure calendar signal — correlation with H001 comes solely from equity risk premium
  exposure (holding SPY ~50% of the time)
