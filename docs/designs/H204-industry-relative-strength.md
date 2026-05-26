# H204: IndustryRelativeStrength

**Signal:** Moskowitz & Grinblatt (1999) — stock return minus sector-average return over 21d
**Date:** 2026-05-25

## Results

**Decision: KEEP**

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.824 |
| 95% CI | [0.421, 1.233] |
| CAGR | 15.43% |
| Max DD | −52.86% |
| H001 corr | +0.004 |

Sensitivity (rel_window): 5d 0.839, 10d 0.808, 21d 0.824, 42d 0.903, 63d 0.996. Longer windows are stronger; 63d nearly at 1.0 Sharpe — the idiosyncratic signal is more reliable at monthly vs weekly frequency. Canonical 21d chosen before OOS.

H001 corr +0.004 — near-zero, confirming sector-demeaned returns are orthogonal to the equity trend signal. Short-horizon idiosyncratic momentum (stock outrunning its sector) is a distinct alpha source from intermediate momentum. All 12 folds positive, exceptional consistency.
