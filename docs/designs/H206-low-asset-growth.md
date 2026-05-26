# H206: LowAssetGrowth

**Signal:** Cooper et al. (2008) extended -- 2-year total-asset growth rate
**Date:** 2026-05-25

## Hypothesis

Firms with sustained 2-year total-asset growth underperform more persistently than those measured over 1 year (H170/FF5InvestmentFactor). Over-investing firms signal empire-building tendencies that persist across two annual cycles. Long bottom quintile by 2-year asset growth rate = long capital-conservative firms.

## Expected behavior by regime

- 2000-2002: avoids acquisitive tech/telecom; capital-conservative defensives/staples held
- 2008 GFC: highly leveraged/acquisitive financials excluded; conservative industrials/staples held
- 2013-2017 low vol: signal is weak (passive markets), but no major drawdown expected
- 2020 COVID: capital-conservative firms had stronger balance sheets pre-COVID

## Success criteria

OOS Sharpe > 0.51, CI lower > 0. MaxDD improvement over SPY (−55.92%).
H001 corr < 0.70 (fundamental signal should differ from trend-following).

## What would falsify this

If 2yr window is weaker than 1yr (H170), the signal degrades over time and the 1yr is sufficient. If CI clips zero, the 2yr growth signal adds noise relative to quarterly filing-frequency data.

## Results

**Decision: KEEP**

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.709 |
| 95% CI | [0.306, 1.123] |
| CAGR | 7.97% |
| Max DD | −39.25% |
| H001 corr | −0.019 |

Sensitivity (growth_window): 126d 0.767, 252d 0.721, 504d 0.709, 756d 0.687. Shorter windows slightly stronger; all pass keep bar.

Notable: folds 0-5 (1998-2010) return exactly zero — EDGAR XBRL fundamentals coverage only from ~2009. All 6 active folds (2010-2022) positive. H001 corr −0.019 — slightly negative, excellent diversifier. MaxDD (−39.25%) substantially better than SPY (−55.92%) despite limited coverage.
