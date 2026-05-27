# H287: InvestmentEfficiencyRatio — Revenue Growth per Unit of Capital Invested

## Hypothesis

Productive investment generates revenue. Empire-building investment does not. The investment
efficiency ratio separates productive from unproductive capital allocation:
revenue_growth / (capex_intensity + ε). Firms with high ratios grow revenue efficiently;
firms with low ratios spend heavily without revenue payoff.

Signal: rev_growth = (revenue - revenue.shift(252)) / abs(revenue.shift(252)).clip(-5, 5).
capex_intensity = lt_asset_change / TA.where(TA > 0) where lt_asset_change = (TA-CA) change.
efficiency = rev_growth / (capex_intensity.clip(lower=0) + 0.01).
Rank descending (highest efficiency = best). Long top-quintile. Sector-neutral, monthly rebalance.

Distinction from H286 (low capex growth): H286 avoids high capex growth. H287 rewards
efficient capex — high revenue growth from modest capital deployment. The ratio is positive
for productive firms and near-zero for empire-builders.

Expected regime behavior:
- **2000–2002**: Internet-era capex without revenue is the textbook empire-building case.
- **2008–2009**: Revenue collapses with capex; ratio noisy. Signal disrupted.
- **2013–2017**: Stable investment cycles; productivity differentiates well.
- **2020**: Revenue shock in Q1-Q2; recovery + capex freeze creates high efficiency in H2.

Success criteria: OOS Sharpe ≥ 0.76, CI lower > 0.
Falsification: Ratio collapses to revenue growth alone (capex denominator adds nothing).

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
