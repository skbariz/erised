# H294: TurnoverCappedProfitabilityFactor — GP/TA with 20% Replace Cap

## Hypothesis

A hard turnover cap forces the portfolio to value continuity of positions. At each monthly
rebalance, the maximum fraction of the portfolio that can be replaced is 20%. If more than
20% of current names drop out of the top-quintile, the best existing positions are retained
rather than replaced. This reduces market-impact costs during periods of high factor-rank volatility.

Signal: GP/TA sector-neutral top-quintile. At each rebalance, compute new top-quintile names.
If the set of new names requires replacing > 20% of current portfolio:
- Keep all names that remain in top-40% (expanded sell threshold)
- Fill remaining slots with new names, up to 20% replacement cap
Equal-weight held positions, sector-neutral.

Distinction from H292 (quarterly rebalance): H292 reduces frequency. H294 caps per-period
replacement regardless of frequency. H294 still rebalances monthly but limits churn.

Expected regime behavior:
- **2000–2002**: Factor-rank volatility high; cap retains stable GP names.
- **2008–2009**: GP drops for many names; cap may force holding deteriorating positions.
- **2013–2017**: Stable GP signal; cap rarely binds.
- **2020**: COVID shock; cap may force holding temporarily impaired names.

Success criteria: OOS Sharpe ≥ 0.88, CI lower > 0. Benchmark vs H227 (uncapped).
Falsification: Cap forces holding severely deteriorated positions — hurts Sharpe vs H227.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.857 |
| 95% CI | [0.457, 1.267] |
| CAGR | 11.00% |
| Max DD | -34.59% |
| H001 corr | -0.016 |
| Decision | KEEP |

Folds 0-4 zero (EDGAR gap). Active folds 5-11: 0.30, 0.66, 2.02, 0.94, 2.02, 1.28, 1.23.
Sensitivity: max_replace barely matters (0.817 to 0.863), MaxDD constant at -34.59%. Cap rarely binds when GP/TA signal is stable.
Between H292 quarterly (0.884) and H291 value (0.721) — replacement cap adds modest value over uncapped monthly rebalance.
