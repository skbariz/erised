# H209: BuybackMomentum

**Signal:** z(-net_issuance_252) + z(12-1m price momentum)
**Date:** 2026-05-25

## Hypothesis

Combining management confidence (buyback = management believes stock is undervalued) with market confirmation (price momentum = market agrees). Both signals should fire together for the strongest candidates. Expected to improve over pure momentum (H150) by excluding momentum stocks where management is diluting shareholders.

## Expected behavior by regime

- 2000-2002: momentum signal fails early, but buyback gate may filter some of the worst crash stocks
- 2008 GFC: momentum fails everywhere; buyback signal disrupted by forced share issuance (bank recaps)
- 2013-2017: ideal environment -- strong momentum AND strong buyback era; should excel
- 2020 COVID: momentum works quickly post-crash; buybacks resume fast in 2021

## Success criteria

OOS Sharpe > 0.65 (should beat H150's 0.981 baseline is aspirational; beating SPY 0.51 is minimum), CI lower > 0.

## What would falsify this

If buyback signal has no incremental value over pure momentum, composite equals H150. If shares_outstanding coverage gaps pre-2009 degrade early folds to pure momentum, the strategy adds cost without benefit.

## Results

**Decision: KEEP**

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.930 |
| 95% CI | [0.527, 1.344] |
| CAGR | 18.03% |
| Max DD | −49.88% |
| H001 corr | +0.002 |

Sensitivity (issuance_window): 126d 0.937, 252d 0.930, 504d 0.933. Remarkably stable — momentum component dominates the signal variance; buyback provides consistent filtering across windows.

Full 12-fold coverage (momentum price data covers pre-2009 when shares data is absent). Sharpe 0.930 — best in Batch 2 by 0.162 over next best (H209 vs H207). H001 corr +0.002 near-zero. MaxDD −49.88% reflects the momentum loading. Top candidate for cross-program ensemble alongside H202/H203.
