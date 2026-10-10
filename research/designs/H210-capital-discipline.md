# H210: CapitalDiscipline

**Signal:** z(-asset_growth) + z(-net_issuance) + z(-leverage_change) composite
**Date:** 2026-05-25

## Hypothesis

Three dimensions of capital allocation discipline:
1. Low asset growth: not expanding capacity without returns
2. Low share issuance: not diluting; preferring buybacks
3. Low leverage increase: Δ(LTD/TA) YoY -- not taking on debt to fund growth

Firms that score high on all three are genuinely disciplined capital allocators. The leverage dimension differentiates from H208 (which uses OCF quality instead of leverage change).

## Expected behavior by regime

- 2000-2002: tech/telecom growth companies both expand assets AND issue shares AND take on debt. Triple signal.
- 2008 GFC: leverage dimension is most diagnostic -- avoid financial firms ramping leverage pre-crisis
- 2013-2017: leverage change signal is neutral (rates low, debt cheap); asset/issuance drive alpha
- 2020 COVID: leverage increases everywhere; signal may flip or weaken

## Success criteria

OOS Sharpe > 0.51, CI lower > 0. Materially different H001 corr from H208 (different third signal).

## What would falsify this

If long_term_debt coverage is sparser than total_assets (banks reclassify; some sectors have LTD = NaN), the leverage component drops out for large fractions of the universe. If coverage < 50%, the composite effectively reduces to H206+H207 (asset growth + issuance only).

## Results

**Decision: KEEP**

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.733 |
| 95% CI | [0.329, 1.141] |
| CAGR | 8.44% |
| Max DD | −37.64% |
| H001 corr | −0.019 |

Sensitivity (window): 126d 0.761, 252d 0.733, 504d 0.722. Shorter windows consistently better.

Folds 0-5 zero (long_term_debt coverage mirrors total_assets — EDGAR gap). MaxDD −37.64% between H207 (−38.07%) and H208 (−35.97%). H001 corr −0.019 confirms capital discipline signals are orthogonal to equity trend. Leverage change signal adds marginal value vs H208 (different third dimension, similar Sharpe/MaxDD).
