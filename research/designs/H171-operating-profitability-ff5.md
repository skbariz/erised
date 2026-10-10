# H171: OperatingProfitabilityFF5

## Hypothesis
Long top quintile by gross profit / book equity (GP/BE), sector-neutral.
Fama-French (2015): RMW (robust minus weak) factor uses operating profitability = (revenue - COGS - SGA - interest) / book equity.
GP/BE is a close approximation available from our EDGAR data.
Distinct from H165 (GP/TA): denominator is book equity, capturing return on equity capital.

## Expected regime behavior
- High GP/BE firms: strong economic moats, pricing power relative to equity capital
- 2000-2002: profitable companies at book value should survive tech crash
- 2008-2009: firms with high GP/BE may be overlevered; BE inflated; mixed
- 2013-2017: growth stocks with high GP/BE should perform well

## Success criteria
- OOS Sharpe > 0.60
- Different from H165 (GP/TA) — if correlation > 0.90, redundant
- H001 corr < 0.75

## What would falsify this
- Near-identical to H165 (GP/TA uses same signal, different scaling)
- Sharpe < 0.51 or CI spans zero
