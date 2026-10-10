# H281: GrossProfitImprovementFactor — YoY Gross Profit Dollar Growth

## Hypothesis

H180 (EarningsTrendLong) tested NI growth as an earnings attention proxy (Sharpe 0.715).
Gross profit improvement is a distinct signal: GP growth captures revenue quality growth
(pricing power and cost control) rather than bottom-line earnings. GP is less susceptible
to accounting manipulation through below-the-line items.

Signal: gp_change = (GP - GP.shift(252)) / abs(GP.shift(252)).clip(-5, 5).
Rank descending. Long top-quintile (largest GP dollar improvement). Sector-neutral, monthly rebalance.

Distinction from existing strategies:
- H180: NI change (bottom-line earnings surprise)
- H266: GP/revenue level (gross margin, not GP dollar growth)
- H267: YoY change in gross margin (GP/rev) — not GP dollar amount
- H268: GP/TA level (knowledge return proxy)

GP dollar growth is distinct from all of these — tests whether top-line cost-efficiency
improvement (gross profit expansion) is independently predictive.

Expected regime behavior:
- **2000–2002**: Tech companies with collapsing revenues generate negative GP growth.
- **2008–2009**: Broad GP contraction; sparse selection.
- **2013–2017**: Steady economic expansion drives consistent GP improvement.
- **2020**: COVID revenue/cost shock; recovery strong in H2 2020.

Success criteria: OOS Sharpe ≥ 0.70, CI lower > 0, H180 corr < 0.85.
Falsification: H180 corr > 0.85 — GP growth and NI growth rank stocks identically.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.775 |
| 95% CI | [0.366, 1.188] |
| CAGR | 9.24% |
| Max DD | -35.70% |
| H001 corr | -0.015 |
| Decision | KEEP |

Folds 0-5 zero (EDGAR data gap). Active folds 6-11: 0.37, 1.92, 0.92, 1.84, 1.21, 1.11.
Flat sensitivity (quantile 0.15-0.25: Sharpe 0.772-0.776) — structural signal.
GP dollar growth is a valid standalone signal at 0.775, weaker than GP/TA level (H268: 0.861).
Consistent with level > change pattern observed in H201-H250.
