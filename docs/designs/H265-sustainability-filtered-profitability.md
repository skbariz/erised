# H265: SustainabilityFilteredProfitability — GP/Assets With Accounting Guard

## Hypothesis

Gross profitability (GP/assets) is a durable signal (H227: 0.878). However, it can select
firms with aggressive revenue recognition or accrual-heavy income. The H264 sustainability
composite filters out accounting-unreliable high-GP firms.

Signal: Compute gross profitability (gross_profit / total_assets). Compute H264 sustainability
score (NOA + accruals + CFO/NI composite). Include only stocks with top-half sustainability
score. Rank remaining by gross profitability; long top-quintile of qualifying names.
Sector-neutral, monthly rebalance.

This is the "clean GP" universe: high gross profitability confirmed by accounting quality.
It avoids the Enron-style case of high reported gross margin backed by aggressive accruals.

Expected regime behavior:
- **2000–2002**: Strong — filters tech firms with inflated GP from dubious revenue recognition.
- **2008–2009**: Good — sustainability filter removes overleveraged financials.
- **2013–2017**: Strong — should perform well as high-quality names dominate.
- **2020**: Mixed — working capital distortions may temporarily penalize quality firms.

Success criteria: OOS Sharpe ≥ 0.92, Max DD < −38%, beats H227 standalone (0.878), CI lower > 0.
Falsification: GP correlation with sustainability composite > 0.85 (filter redundant).

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
