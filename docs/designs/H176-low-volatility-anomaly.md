# H176: LowVolatilityAnomaly

## Hypothesis
Low-volatility stocks earn higher risk-adjusted returns than high-volatility stocks (Baker, Bradley & Wurgler 2011). The anomaly is driven by investor preference for lottery-like payoffs (overweighting high-vol/high-skew stocks) and benchmark constraints. A sector-neutral portfolio of the lowest-realized-volatility 20% of stocks within each GICS sector should capture this. Realized volatility is computed from 252-day rolling standard deviation of daily returns — always computable from price data, no data gaps.

Compared to BAB (H175), this uses total realized volatility rather than systematic (market) beta. The anomaly may be distinct: low-vol stocks are not always low-beta (idiosyncratic vol contributes).

## Expected behavior
- **2000-2002 tech crash:** High-vol tech stocks implode; low-vol stocks defensive — strong alpha
- **2008-2009 GFC:** Low-vol stocks still fall but less; expect weak positive or flat fold
- **2020 COVID:** Same as GFC — modest performance
- **Low-vol 2013-2017:** Bull market rewards high-vol; low-vol lags; Sharpe < 1.0 likely

## Success criteria
- OOS Sharpe > 0.51 with CI lower bound > 0
- All 12 folds positive (price-only signal, no data gaps)
- H001 corr in 0.50–0.75 range

## What would falsify this
- Negative folds in 2000-2002 (should be low-vol's best period)
- Lower Sharpe than BAB (H175) — would suggest beta is the cleaner signal

## Results (OOS)
- **Decision: KEEP**
- Sharpe 0.877, CI [0.466, 1.293], CAGR 13.35%, MaxDD -40.48%
- H001 corr: +0.6604
- All 12 folds positive (min: +0.066 in 2008-2010)
- Sensitivity: vol_window 63/126/252 all near 0.88 — robust
- Lower Sharpe than BAB (1.088) — beta is a cleaner signal than total vol in this universe
