# H169: QualityValueComposite

## Hypothesis
(1/3)*z(B/M) + (1/3)*z(GP/A) + (1/3)*z(-accruals), sector-neutral.
Combines three distinct quality signals:
- Value: book-to-market (Fama-French HML, cheap stocks)
- Quality: gross profitability (Novy-Marx, productive stocks)
- Earnings quality: low accruals (Sloan, cash-driven earnings)

Intuition: these three signals are partially orthogonal — combining them should reduce
noise while preserving alpha from each component.

## Expected regime behavior
- 2000-2002: cheap stocks that are also profitable and cash-driven — survives tech crash
- 2008-2009: cheap financial stocks are cheap for a reason; GP/A filter may exclude them
- 2013-2017: growth-dominated; value component may drag but GP/A and accruals should compensate

## Success criteria
- OOS Sharpe > 0.70 (expect better than single signals due to diversification)
- CI lower bound > 0
- H001 corr < 0.70

## What would falsify this
- Sharpe below individual component strategies (no diversification benefit)
- Value drag overwhelming quality signals post-2010

## Results (2026-05-25)
- Sharpe: 0.839  CI: [0.433, 1.250]  CAGR: 10.32%  MaxDD: -37.11%
- H001 corr: +0.6281
- Decision: **KEEP**
- Notes: 5 zero folds (pre-EDGAR). Post-2009: 0.78–2.16. Robust sensitivity.
  Better MaxDD than H160 (pure value: -47.26%) — quality signals reduce drawdown.
  within_sector_quantile=10% improves Sharpe to 0.898.
