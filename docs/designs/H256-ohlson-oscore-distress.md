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
| OOS Sharpe | 0.766 |
| 95% CI | [0.369, 1.167] |
| CAGR | 9.51% |
| Max DD | −36.97% |
| H001 corr | −0.018 |
| Decision | **KEEP** |

**Sensitivity (within_sector_quantile):** 10% → 0.773 | 20% → 0.766 | 30% → 0.761. Tighter selection marginally better; flat sensitivity confirms structural signal. Canonical = 20%.

**Folds 0–4 zero:** Fundamental data unavailable before ~2008 train window. Effective 7-fold OOS over 2008–2022. All 7 active folds positive (lowest +0.367, highest +2.016). H001 corr −0.018 is essentially zero — this family is completely orthogonal to trend-following signals.
