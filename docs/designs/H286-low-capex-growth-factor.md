# H286: LowCapexGrowthFactor — Capex Proxy Growth (LT Asset Change)

## Hypothesis

Titman, Wei & Xie (2004) show that firms that invest more heavily underperform — empire-building
management overestimates project NPV. We test the capex-specific investment signal. Since
`capex` is unavailable in the data schema, we use the long-term asset change proxy:
lt_assets = (TA - CA), delta_lt = (lt_assets - lt_assets.shift(252)) / TA.

This is the same proxy used in H263 (ConservativeCapexInvestment), but here we test it as a
YoY capex-GROWTH signal (percentage change) rather than level.

Signal: capex_growth = (delta_lt - delta_lt.shift(252)) / abs(delta_lt.shift(252)).clip(-10, 10).
Rank ascending (low capex growth = disciplined). Long bottom-quintile. Sector-neutral, monthly rebalance.

Distinction from H263 (ConservativeCapexInvestment): H263 uses the level of LT asset change
(how much was invested). H286 uses the acceleration (how much faster investment is growing).
Acceleration above baseline is the warning signal for empire-building.

Expected regime behavior:
- **2000–2002**: High capex growth in 1999-2000 leads to poor 2001-2002 returns.
- **2008–2009**: Capex slashed broadly; signal less discriminating.
- **2013–2017**: Normal investment cycle; steady signal.
- **2020**: COVID capex freeze creates V-shape; signal disrupted temporarily.

Success criteria: OOS Sharpe ≥ 0.72, CI lower > 0, H263 corr < 0.85.
Falsification: H263 corr > 0.85 — capex growth is just lagged capex level.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
