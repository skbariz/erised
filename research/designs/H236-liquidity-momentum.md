# H236: LiquidityMomentum — YoY Improvement in Current Ratio

## Hypothesis

Companies with improving short-term liquidity (rising current ratio = CA/CL) face lower financial distress risk going forward. An improving current ratio signals that management is building a buffer against obligations — either by accumulating assets or paying down liabilities. This should predict positive abnormal returns as distress risk reprices.

Expected regime behavior:
- **2000–2002 tech crash**: Improving liquidity should be protective — firms building cash buffers survive better.
- **2008–2009 GFC**: Strong signal expected; credit crunch punishes low-liquidity firms severely.
- **2020 COVID**: Mixed — many firms drew credit lines (inflating CL) regardless of underlying quality.
- **2013–2017 low-vol**: Moderate — distress risk is low overall; signal may underperform.

Success criteria: OOS Sharpe > 0.51, CI lower bound > 0, H001 corr < 0.10.
Falsification: If improving current ratio has no predictive power, folds 6-11 would show Sharpe ≈ 0.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.815 |
| 95% CI | [0.404, 1.229] |
| CAGR | 9.51% |
| Max DD | −36.93% |
| H001 corr | −0.016 |
| Decision | **KEEP** |

**Sensitivity (window):** 126d → 0.791, 252d → 0.815, 504d → 0.749. Flat profile; canonical 252d is strongest.

**Folds 6–11 Sharpe:** 0.682, 2.140, 0.929, 2.029, 1.185, 0.947. Folds 0–5 zero (EDGAR coverage starts ~2009).

Signal works robustly in the EDGAR era. Fold 7 (2012–2013) and fold 9 (2015–2017) are exceptional. Near-zero H001 correlation confirms pure fundamental signal with no price contamination.
