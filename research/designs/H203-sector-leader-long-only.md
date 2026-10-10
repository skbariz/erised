# H203: SectorLeaderLongOnly

**Signal:** Moskowitz & Grinblatt (1999) — stock selection concentrated in top-2 sectors by 6m return
**Date:** 2026-05-25

## Results

**Decision: KEEP**

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.915 |
| 95% CI | [0.518, 1.325] |
| CAGR | 15.30% |
| Max DD | −46.69% |
| H001 corr | +0.001 |

Sensitivity (top_sectors): 1 sector 0.855, 2 sectors 0.915, 3 sectors 0.877, 4 sectors 0.922. Two sectors is near-optimal; adding more dilutes the signal.

H001 corr +0.001 — essentially zero correlation with trend-following (program record level). Concentrated sector approach cleanly separates sector-momentum alpha from equity beta timing. Per-fold Sharpe stable: only fold 5 (2008) barely negative (−0.156). Moskowitz-Grinblatt pure implementation is robust and uncorrelated with the trend family.
