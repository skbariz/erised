# H211: BalanceSheetAccruals

**Signal:** Sloan (1996) balance-sheet approach: delta(NWC) / avg_TA
**Date:** 2026-05-25

## Hypothesis

Balance-sheet accruals = Δ(current_assets - current_liabilities) / avg_TA. Distinct from H166 (LowAccrualsQuality) which uses (NI-OCF)/TA cash-flow approach. NWC growth signals channel stuffing, inventory build, or aggressive revenue recognition -- all signs of lower earnings quality predicting underperformance. Sloan (1996) original formulation.

## Results

**Decision: KEEP**

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.800 |
| 95% CI | [0.396, 1.206] |
| CAGR | 9.22% |
| Max DD | −37.35% |
| H001 corr | −0.021 |

Sensitivity (accrual_window): 126d 0.769, 252d 0.800, 504d 0.699. 252d optimal.
Folds 0-5 zero (EDGAR ~2009+). All 6 active folds positive. H001 corr −0.021 — negative correlation. MaxDD −37.35% — similar to H208. Balance-sheet approach validated alongside H166 (cash-flow approach, Sharpe 0.824). Two independent accruals definitions both predict underperformance.
