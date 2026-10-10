# H175: BettingAgainstBeta

## Hypothesis
Low-beta stocks earn higher risk-adjusted returns than high-beta stocks (Frazzini & Pedersen 2014). Constrained investors (mutual funds, pension funds) are forced to hold high-beta stocks to generate market-like returns, inflating their prices. Low-beta stocks are therefore underpriced relative to their systematic risk. A sector-neutral long-only portfolio of the lowest-beta 20% of stocks within each GICS sector should capture this anomaly while avoiding sector concentration.

Beta is estimated via 252-day rolling OLS regression of daily stock returns on SPY returns. Sector-neutrality prevents the strategy from simply picking defensive sectors (Utilities, Consumer Staples).

## Expected behavior
- **2000-2002 tech crash:** High-beta tech stocks crushed; low-beta stocks should hold up well — strong alpha expected
- **2008-2009 GFC:** Universal panic selling; low-beta stocks still fall but less. Expect positive fold Sharpe
- **2020 COVID:** Sharp crash + rapid recovery; low-beta stocks lag on the way up. Mild underperformance likely
- **Low-vol 2013-2017:** Bull market with low dispersion; low-beta stocks underperform high-beta. Modest Sharpe

## Success criteria
- OOS Sharpe > 0.51 with CI lower bound > 0
- All 12 folds positive (price-based beta is always computable, no data gaps)
- H001 corr in range 0.50–0.80 (partially overlaps trend-following, but different mechanism)

## What would falsify this
- Negative or near-zero folds in 2000-2002 or 2008-2009 (should be BAB's best periods)
- CI lower bound <= 0 (noise, not signal)
- Sharpe worse than plain momentum baseline

## Results (OOS)
- **Decision: KEEP**
- Sharpe 1.088, CI [0.681, 1.497], CAGR 16.32%, MaxDD -40.75%
- H001 corr: +0.6384
- All 12 folds positive (min: +0.123 in 2008-2010)
- Sensitivity: beta_window 126/252/504 all near 1.09 — robust
- Beats H150 plain momentum baseline (0.981) by +0.107 Sharpe
