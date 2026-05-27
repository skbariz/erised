# H291: LowTurnoverValueFactor — Earnings Yield with Buy/Hold Band

## Hypothesis

Novy-Marx & Velikov (2016) show that value factors have heavy rank churn that destroys net alpha.
A buy/hold band — buy on entry into top-quintile, sell only on exit below top-40% — reduces
round-trip turnover substantially while maintaining similar gross exposure.

Signal: earnings_yield = NI / (price × shares).where(price × shares > 0) (earnings/market cap).
Sector-ranked, monthly. Buy rule: enter position when earnings_yield rank ≥ 80th percentile.
Hold rule: maintain position until earnings_yield rank < 60th percentile (sell only on exit
below top-40%).

Implementation: track position state; new buys only at rebalance when rank ≥ 0.80; forced
sells when rank < 0.60. All others held. Equal-weight held positions, sector-neutral.

Distinction from standard earnings yield: Standard monthly rebalance replaces ~40% of portfolio
monthly. Buy/hold band reduces this to ~15%, cutting round-trip costs by ~62%.

Expected regime behavior:
- **2000–2002**: Value anomaly strong; band retains value names through noise.
- **2008–2009**: Earnings collapse; band forces sell at 0.60 threshold protecting from traps.
- **2013–2017**: Stable earnings; low turnover. Band reduces churn substantially.
- **2020**: COVID shock forces sells at 0.60 threshold.

Success criteria: OOS Sharpe ≥ 0.75, CI lower > 0.
Falsification: Band causes portfolio to drift into value traps held past deterioration.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
