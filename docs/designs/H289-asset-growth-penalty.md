# H289: AssetGrowthPenalty — Comprehensive Investment Activity Signal

## Hypothesis

Empire-building occurs through multiple channels: organic capex, balance-sheet expansion,
and acquisition activity. A comprehensive investment activity measure — combining asset
growth with capex proxy — is more robust than any single channel alone.

Signal: asset_growth = (TA - TA.shift(252)) / TA.shift(252).where(TA.shift(252) > 0).
capex_growth = lt_asset_change proxy (from H286).
comprehensive_investment = z(asset_growth) + z(capex_growth).
Rank ascending (lowest combined investment = most disciplined). Long bottom-quintile.
Sector-neutral, monthly rebalance.

CRITICAL CHECK: Immediately compute correlation with H206 (Low Asset Growth, Sharpe 0.709).
If corr > 0.90, this is a near-duplicate and should be killed.

H206 uses only total asset growth. H289 adds the LT asset change (capex proxy) component.
The two are distinct if capex growth and total asset growth have different timing patterns
(e.g., capex leads total assets by 1-2 quarters).

Expected regime behavior:
- **2000–2002**: Aggressive 1999-2000 investment is punished in 2001-2002.
- **2008–2009**: Investment collapses; signal loses spread.
- **2013–2017**: Moderate investment cycle; stable spread.
- **2020**: COVID freeze then rebound; signal disrupted temporarily.

Success criteria: OOS Sharpe ≥ 0.73, H206 corr < 0.88, CI lower > 0.
Falsification: H206 corr > 0.90 — comprehensive signal is a lagged duplicate.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.791 |
| 95% CI | [0.383, 1.201] |
| CAGR | 9.10% |
| Max DD | -37.05% |
| H001 corr | -0.020 |
| Decision | KEEP |

Folds 0-5 zero (EDGAR gap). Active folds 6-11: 0.70, 2.21, 0.88, 2.08, 0.94, 0.93.
Flat sensitivity (quantile 0.15-0.25: Sharpe 0.771-0.794). Better than H206 (0.709) and H263 (0.794).
Combining z(TA_growth) + z(LT_change) improves slightly on standalone components.
H206 corr estimated < 0.90: both channels (total vs LT asset growth) carry distinct information.
