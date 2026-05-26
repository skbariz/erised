# H214: GrossMarginMomentum

**Signal:** 12-1m momentum gated by YoY improving gross margin
**Date:** 2026-05-25

## Hypothesis

Only buy momentum stocks where gross_profit/revenue has improved YoY (expanding margins = pricing power). Filters momentum to companies with genuine competitive advantage improvement.

## Results

**Decision: KEEP**

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.690 |
| 95% CI | [0.282, 1.099] |
| CAGR | 8.52% |
| Max DD | −34.55% |
| H001 corr | −0.014 |

Sensitivity (margin_window): 126d 0.803 / −33.50%, 252d 0.690 / −34.55%, 504d 0.907 / −28.87%. Longer windows substantially better: 504d shows Sharpe 0.907 and MaxDD −28.87%. Canonical 252d chosen before OOS.

Best MaxDD in batch at 252d canonical (−34.55%). H001 corr −0.014. Gross margin improvement is a high-quality signal but the 504d window suggests slower-moving margin expansion is more predictive. Folds 0-5 zero (EDGAR gap); all 6 active folds with fold 6 slightly negative (−0.118).
