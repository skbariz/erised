# H213: EarningsQualityMomentum

**Signal:** z(-balance_sheet_accruals) + z(12-1m price momentum)
**Date:** 2026-05-25

## Hypothesis

Composite: Sloan (1996) balance-sheet accruals quality + Jegadeesh-Titman intermediate momentum. The momentum component covers periods before fundamental data availability (pre-2009), giving full 12-fold coverage.

## Results

**Decision: KEEP** (Batch 3 star)

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.907 |
| 95% CI | [0.496, 1.316] |
| CAGR | 17.43% |
| Max DD | −49.88% |
| H001 corr | +0.001 |

Sensitivity (accrual_window): 126d 0.923, 252d 0.907, 504d 0.923. All windows strong and stable.
Full 12-fold coverage. H001 corr +0.001 -- essentially zero. Composite z-score approach better than AND condition (H212): averaging signals captures more breadth. Same MaxDD as pure momentum (H150's momentum component dominates early folds); quality adds stability in active folds.
