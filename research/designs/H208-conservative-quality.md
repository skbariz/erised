# H208: ConservativeQuality

**Signal:** 3-signal capital discipline composite: z(-asset_growth) + z(-net_issuance) + z(OCF/TA)
**Date:** 2026-05-25

## Hypothesis

Three independent capital discipline signals averaged via cross-sectional z-scores:
1. Low asset growth (Cooper et al. 2008): avoid empire-builders
2. Low net share issuance (Pontiff-Woodgate 2008): avoid diluters / prefer buybacks
3. High OCF/TA: Sloan (1996) cash quality signal

The composite should be more robust than any individual signal because the three dimensions are partially independent. Expected Sharpe lift over H206/H207 individually.

## Expected behavior by regime

- 2000-2002: multiple signals fire simultaneously against tech growth/issuance
- 2008 GFC: OCF signal weakens (earnings decline) but asset + issuance gate protects
- 2013-2017: buyback era strengthens the issuance signal

## Success criteria

OOS Sharpe > 0.60 (composite should exceed H206/H207 individually), CI lower > 0.

## What would falsify this

If OCF coverage is too sparse pre-2009 (similar to accruals in H166), the composite degrades to a 2-signal model and loses the diversification benefit. If composite Sharpe < max(H206, H207), the signals cancel rather than diversify.

## Results

**Decision: KEEP**

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.762 |
| 95% CI | [0.362, 1.174] |
| CAGR | 9.19% |
| Max DD | −35.97% |
| H001 corr | −0.020 |

Sensitivity (window): 126d 0.798, 252d 0.762, 504d 0.775. All windows robust.

OCF coverage enabled fold 5 (2008-2010) to be positive (+0.519) unlike pure H206/H207 (zero). MaxDD −35.97% best in batch — composite averaging reduces extremes. Three-signal diversification validates the hypothesis: partial independence between asset growth, issuance, and OCF quality reduces fold variance.
