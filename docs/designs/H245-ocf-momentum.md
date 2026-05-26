# H245: OCFMomentum — YoY Improvement in OCF/Total Assets

## Hypothesis

Improving cash return on assets (OCF/TA trend) captures real operational improvement at the cash level. Unlike ROA momentum (H240) which uses NI, OCF/TA avoids accrual distortions — improving OCF/TA means genuinely better cash generation from the same asset base. The YoY change version captures momentum in cash efficiency.

Signal: (OCF/TA)_t − (OCF/TA)_{t−1y}. Sector-neutral top 20%, monthly rebalance.

Expected regime behavior:
- **2000–2002 tech crash**: OCF-improving companies are genuinely operationally improving; signal avoids cash-burning tech.
- **2008–2009 GFC**: OCF deterioration hits cyclicals; relative winners are identified.
- **2020 COVID**: Rapid OCF changes; quality of the signal tested.
- **2013–2017 low-vol**: Steady improvement premium expected.

Success criteria: OOS Sharpe > 0.51, CI lower bound > 0. Expect slightly weaker than ROA (H240) due to OCF volatility, but higher quality signal.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.741 |
| 95% CI | [0.334, 1.146] |
| CAGR | 8.64% |
| Max DD | −37.99% |
| H001 corr | −0.018 |
| Decision | **KEEP** |

**Sensitivity (window):** 126d → 0.757 | 252d → 0.741 | 504d → 0.742. Essentially flat — window doesn't matter; OCF/TA trend is slow-moving.

**Folds 6–11 Sharpe:** 0.437, 1.923, 0.919, 1.831, 0.938, 1.071. Folds 0–5 zero.

Consistent with ROA momentum (H240: 0.751) with comparable MaxDD, confirming that the accrual adjustment adds limited incremental value in this dataset (EDGAR reporting quality is generally high). Both pass cleanly.
