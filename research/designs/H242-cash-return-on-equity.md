# H242: CashReturnOnEquity — OCF / Book Equity

## Hypothesis

Cash return on equity (OCF / book_equity) is the cash-flow analog to ROE. Unlike NI-based ROE, OCF cannot be manipulated by accrual choices, making it a purer measure of how much real cash each dollar of equity generates. High cash ROE should predict positive abnormal returns as the market recognizes genuine cash generation capacity.

Signal: OCF / BE where BE > 0, clipped [-1.0, 5.0]. Sector-neutral top 20%, monthly rebalance.

Expected regime behavior:
- **2000–2002 tech crash**: Real cash generators survive; the signal identifies them.
- **2008–2009 GFC**: Cash-rich equity bases are protective. Signal should be active in fold 5.
- **2020 COVID**: Bifurcation favors cash-generative firms; expected fold performance.
- **2013–2017 low-vol**: Quality premium; moderate positive performance.

Success criteria: OOS Sharpe > 0.51, CI lower bound > 0. Distinct from ROE momentum (H221) and OCF/TA (H245).

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.766 |
| 95% CI | [0.365, 1.176] |
| CAGR | 9.32% |
| Max DD | −38.86% |
| H001 corr | −0.021 |
| Decision | **KEEP** |

**Sensitivity (within_sector_quantile):** 10% → 0.810 | 20% → 0.766 | 30% → 0.760. Mild concentration benefit at 10%.

**Folds 5–11 Sharpe:** 0.450, 0.745, 2.103, 0.701, 1.867, 0.884, 1.035. Fold 5 shows 0.450 (partial EDGAR coverage).

Clean result. Cash ROE (0.766) slightly outperforms NI-based ROA momentum (H240: 0.751), consistent with the accrual-quality hypothesis.
