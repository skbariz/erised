# H165: GrossProfitabilityRank

## Hypothesis
Long top quintile by gross profit / total assets (GP/A) within each GICS sector.
Novy-Marx (2013): GP/A is the most powerful predictor of expected returns — more robust than
value signals because it measures actual economic productivity (pricing power) rather than
accounting book values. Sector-neutral to avoid betting on productive vs capital-intensive sectors.

## Expected regime behavior
- 2000-2002 tech crash: high GP/A tech firms (Microsoft, Oracle) should outperform speculative growth
- 2008-2009 GFC: quality stocks typically outperform in distress — moderate protection expected
- 2020 COVID: defensive quality firms should hold up well
- 2013-2017 low vol: consistent positive exposure to profitable firms

## Success criteria
- OOS Sharpe > 0.51 AND CI lower bound > 0
- H001 correlation < 0.80 (orthogonal to pure price momentum)

## What would falsify this
- CI spans zero (noise only)
- Post-2009 folds all near zero (XBRL data not providing edge beyond the benchmark)
- Sensitivity shows Sharpe degrades rapidly with wider quantile

## Results (2026-05-25)
- Sharpe: 0.879  CI: [0.469, 1.287]  CAGR: 11.03%  MaxDD: -34.46%
- H001 corr: +0.6199
- Decision: **KEEP**
- Notes: 5 zero folds (pre-EDGAR). Post-2009 performance strong and consistent (0.49–2.06).
  within_sector_quantile=10% improves to Sharpe 0.922 — but canonical 20% chosen.
