# H263: ConservativeCapexInvestment — Net Long-Term Asset Growth Discipline

## Hypothesis

Titman, Wei, and Xie (2004) show that aggressive investment growth predicts lower returns —
firms over-investing destroy value through empire-building. We proxy capex discipline via
YoY change in long-term (non-current) asset base.

Note: `capex` is NOT in VALID_FIELDS. Proxy: net long-term investment =
  lt_assets_t = total_assets_t - current_assets_t
  delta_lt_assets = (lt_assets_t - lt_assets_{t-252}) / total_assets_t  (annual change)

This captures capital deployment beyond working capital — includes PP&E growth and
long-term investment spending, which is the primary capex driver.

Rank ascending (low delta_lt_assets = conservative investment = best).
Sector-neutral top-quintile, monthly rebalance.

Expected regime behavior:
- **2000–2002**: Strong — aggressive tech/telecom infrastructure build-out.
- **2008–2009**: Moderate — investment freezes during crisis; low delta cross-sectional spread.
- **2013–2017**: Moderate — energy/industrial capex expansion punished.
- **2020**: Neutral — capex cuts universal during COVID; spread collapses.

Success criteria: OOS Sharpe ≥ 0.68, CI lower bound > 0.
Falsification: High correlation with asset growth (H206) — same signal, different measurement window.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
