# H262: AccrualReliabilityFactor — Cash-Backed Earnings

## Hypothesis

Richardson, Sloan, Soliman, and Tuna (2005) show that the accrual component of earnings
negatively predicts future returns. Total accruals = net income − cash from operations.
High accruals mean earnings were not cash-backed — less reliable and less persistent.

Signal: total_accruals = (net_income - operating_cf) / avg(total_assets)
  where avg(total_assets) = (total_assets_t + total_assets_{t-252}) / 2 (annual avg proxy)

Rank ascending (low accruals = earnings are cash-backed = best). Sector-neutral top-quintile,
monthly rebalance.

Distinction from H166 (Low Accruals Quality): H166 used a simpler balance-sheet accrual proxy.
H262 uses the direct cash-flow-statement measure (NI − CFO) which is the gold standard in
the Richardson et al. framework.

Expected regime behavior:
- **2000–2002**: Strong — many tech companies had high accruals from aggressive revenue recognition.
- **2008–2009**: Moderate — financials' off-balance-sheet accruals; harder to capture.
- **2013–2017**: Moderate signal strength in normal corporate conditions.
- **2020**: Neutral to positive — cash-generating firms protected during liquidity stress.

Success criteria: OOS Sharpe ≥ 0.72, CI lower bound > 0.
Falsification: Correlation with H166 > 0.90 (redundant proxy).

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
