# H256: OhlsonOScoreDistress — Accounting Distress Avoidance

## Hypothesis

Ohlson (1980) shows that a small set of accounting ratios predicts corporate bankruptcy.
Even outside actual bankruptcy, high O-score stocks earn anomalously low returns because
distress risk is not fully priced and investors prefer lottery-like distress payoffs.

Signal: Proxy O-score from quarterly fundamentals using components available in VALID_FIELDS:
- leverage = (total_assets - book_equity) / total_assets
- working_capital_ratio = (current_assets - current_liabilities) / total_assets
- current_ratio = current_assets / current_liabilities
- roa = net_income / total_assets
- cfo_leverage = operating_cf / (total_assets - book_equity + 1)  [liabilities proxy]

z-score each component, orient so high = distress, sum → composite O-score proxy.
Long bottom-quintile (low distress). Sector-neutral, monthly rebalance.

Expected regime behavior:
- **2000–2002**: Strong — over-leveraged tech names elevated O-score.
- **2008–2009**: Strong — credit crisis reveals accounting distress; signal should spike.
- **2013–2017**: Moderate — low-distress era compresses spread.
- **2020 COVID**: Mixed — sudden revenue collapse floods all sectors with distress signals.

Success criteria: OOS Sharpe ≥ 0.65, Max DD < −45%, CI lower bound > 0.
Falsification: Correlation with H208 (ROE quality) > 0.85 implies not an independent signal.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
