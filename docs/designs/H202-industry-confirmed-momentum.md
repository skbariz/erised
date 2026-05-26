# H202: IndustryConfirmedMomentum

**Signal:** Moskowitz & Grinblatt (1999) — momentum gated by sector 6m trend
**Date:** 2026-05-25

## Results

**Decision: KEEP**

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.908 |
| 95% CI | [0.507, 1.313] |
| CAGR | 17.06% |
| Max DD | −47.16% |
| H001 corr | +0.005 |

Sensitivity (sector_mom_window): 63d 0.934, 126d 0.908, 189d 0.898, 252d 0.980. Robust across all windows; 252d slightly superior.

H001 corr +0.005 — near-zero, among program's lowest. Strong per-fold consistency: only fold 5 (2008 GFC) negative (−0.174); 11/12 positive. Moskowitz-Grinblatt gate is effective: excluding stocks in sectors with negative 6m trend improves Sharpe over raw H150 momentum, with better MaxDD. Key diversifier candidate for cross-program ensemble.
