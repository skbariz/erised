# H249: FinancialHealthComposite — 4-Component Solvency Composite

## Hypothesis

Combining four distinct financial health dimensions in a single composite should reduce idiosyncratic noise while preserving the common "financially strong" signal: cash generation (OCF/TA), low leverage (-LTD/TA), short-term liquidity (current ratio), and profitability (NI/TA). Each z-scored and summed equally.

Signal: z(OCF/TA) + z(-LTD/TA) + z(current_ratio) + z(NI/TA). Sector-neutral top 20%, monthly rebalance.

Expected regime behavior:
- **2000–2002 tech crash**: Pre-revenue firms score poorly on NI/TA; signal avoids them.
- **2008–2009 GFC**: Liquidity and leverage signals are most activated; composite should excel.
- **2020 COVID**: Broad health screening should identify resilient firms across sectors.
- **2013–2017 low-vol**: Quality premium exists but compressed.

Success criteria: OOS Sharpe > 0.51, CI lower bound > 0.
Risk: Including NI/TA (accrual-based) and current ratio alongside OCF signals may dilute the stronger signals.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.588 |
| 95% CI | [0.180, 1.013] |
| CAGR | 7.71% |
| Max DD | −35.73% |
| H001 corr | −0.019 |
| Decision | **KEEP** |

**Sensitivity (within_sector_quantile):** 10% → 0.585 | 20% → 0.588 | 30% → 0.605. Very flat; canonical 20% is midrange.

**Folds 5–11 Sharpe:** −0.232, 0.366, 2.076, 0.646, 1.848, 0.945, 0.995. Fold 5 negative (−0.232) is unusual.

Lowest Sharpe in Batch 10 (0.588) — barely passes the keep bar. The 4-component composite is diluted by component disagreement. Fold 5 negative Sharpe (GFC period) reveals that including current_ratio and NI/TA weakens the OCF-dominant signal during stress. The AltmanZProxy (H244: 0.764) with a similar number of components outperforms significantly.
