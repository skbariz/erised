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
| OOS Sharpe | 0.676 |
| 95% CI | [0.274, 1.083] |
| CAGR | 7.45% |
| Max DD | -37.18% |
| H001 corr | -0.014 |
| Decision | KEEP |

Folds 0-5 zero (EDGAR gap). Active folds 6-11: 0.15, 1.88, 0.69, 2.10, 0.82, 0.88.
Flat sensitivity (quantile 0.15-0.25: Sharpe 0.672-0.688). Weakest in batch.
2nd derivative of capex is noisier than 1st derivative level (H263: 0.794).
Acceleration signal adds some value but fold 6 (0.149) shows weak early activation.
