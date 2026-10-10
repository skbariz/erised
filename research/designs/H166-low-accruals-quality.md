# H166: LowAccrualsQuality

## Hypothesis
Long bottom quintile by accruals = (net_income - operating_cf) / |total_assets|.
Sloan (1996): low accruals signal cash-driven earnings (high earnings quality).
High accruals = earnings inflated by accounting estimates → mean-reversion → lower future returns.
No sector neutralization — accruals are relatively comparable across sectors.

## Expected regime behavior
- 2000-2002 tech crash: aggressive revenue recognition was rampant in tech; low-accrual firms avoided
- 2008-2009 GFC: cash-generating firms more resilient in credit crunch
- 2020 COVID: firms with strong OCF should outperform accrual-heavy firms
- General: signal decays over holding periods > 1 year as restatements occur

## Success criteria
- OOS Sharpe > 0.51 AND CI lower bound > 0
- Low accruals should outperform high accruals in most post-2009 folds

## What would falsify this
- CI spans zero in post-2009 folds (low accruals doesn't predict after widespread awareness)
- Accruals anomaly arbitraged away in modern markets

## Results (2026-05-25)
- Sharpe: 0.824  CI: [0.425, 1.238]  CAGR: 10.54%  MaxDD: -34.52%
- H001 corr: +0.6336
- Decision: **KEEP**
- Notes: 5 zero folds (pre-EDGAR). Post-2009: 0.13–2.10. Robust to top_quantile changes.
  top_quantile=10% is slightly better (Sharpe 0.858) but not materially different.
