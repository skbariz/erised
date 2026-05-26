# H207: ShareholderYield

**Signal:** Net share count reduction (buyback signal) -- shares_outstanding YoY change
**Date:** 2026-05-25

## Hypothesis

Net share issuance = (shares_t - shares_{t-252}) / shares_{t-252}. Stocks with the most share count reduction (most negative net issuance = active buyback) should outperform. Mechanism: management has informational advantage; buybacks signal undervaluation and capital discipline.

## Expected behavior by regime

- 2000-2002: companies conserving cash (not diluting) fare better; tech IPO/secondary issuance collapses
- 2008 GFC: buybacks dry up -- signal may be noisy in crisis; moderate drawdown expected
- 2013-2017: buyback era: signal should be strong as S&P 500 companies run large buyback programs
- 2020 COVID: buybacks suspended broadly; signal quality degrades temporarily

## Success criteria

OOS Sharpe > 0.51, CI lower > 0.
H001 corr < 0.65 (buyback signal is balance-sheet-driven, not price-momentum).

## What would falsify this

If shares_outstanding data coverage is insufficient pre-2009 (EDGAR XBRL gaps), the early folds will have near-zero signal coverage and yield near-zero returns, dragging Sharpe below keep bar. Also: if market-timing of buybacks is so random that YoY change is uncorrelated with future returns.

## Results

**Decision: KEEP**

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.768 |
| 95% CI | [0.360, 1.170] |
| CAGR | 8.92% |
| Max DD | −38.07% |
| H001 corr | −0.022 |

Sensitivity (issuance_window): 126d 0.790, 252d 0.768, 504d 0.717. Shorter windows slightly superior; 252d canonical is robust.

Folds 0-5 zero (EDGAR coverage gap); all 6 active folds positive. H001 corr −0.022 — negative correlation with equity trend, exceptional diversifier. MaxDD −38.07% best in this batch. Buyback signal is clean: quarterly filing cadence provides sufficient resolution for annual measurement.
