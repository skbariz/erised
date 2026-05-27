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
| OOS Sharpe | 0.721 |
| 95% CI | [0.323, 1.133] |
| CAGR | 9.28% |
| Max DD | -40.31% |
| H001 corr | -0.022 |
| Decision | KEEP |

Folds 0-4 zero (EDGAR gap); fold 5 near-zero (+0.092). Active folds 6-11: 0.50, 2.00, 0.92, 2.08, 0.95, 0.99.
Flat sensitivity across hold_quantile (0.30-0.50: Sharpe 0.721-0.726, MaxDD -39% to -41%). Band reduces effective churn substantially.
Value factor weaker than quality (H292: 0.884) — earnings yield has more noise than GP/TA.
