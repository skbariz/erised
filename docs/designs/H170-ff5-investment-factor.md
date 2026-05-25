# H170: FF5InvestmentFactor

## Hypothesis
Long bottom quintile by 1-year total asset growth rate (ΔTA/TA), sector-neutral.
Cooper et al. (2008): firms that grow their asset base aggressively underperform those that don't.
Fama-French (2015): the CMA (conservative minus aggressive investment) factor captures this anomaly.
Low asset growth = conservative investing = higher expected return (fewer empire-building decisions).

## Expected regime behavior
- 2000-2002: aggressive asset growers (tech capex bubble) should underperform
- 2008-2009: companies with low debt/asset growth should be more resilient
- 2013-2017: growth environment; aggressive firms may outperform here (risk)
- General: slowdown in asset expansion predicts mean reversion to profitability

## Success criteria
- OOS Sharpe > 0.51 AND CI lower bound > 0
- H001 corr < 0.80

## What would falsify this
- No difference between low/high asset growth quintiles (investment effect absent in S&P 500)
- EDGAR coverage limits too many folds to zero (growth window = 252d requires prior-year TA)
