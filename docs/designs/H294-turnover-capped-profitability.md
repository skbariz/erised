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
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
