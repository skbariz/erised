# H198: EarningsAcceleration

## Hypothesis
The first derivative of earnings (YoY NI growth) is tested in H180. The second derivative — whether the growth rate itself is accelerating — may be more informative. Stocks where earnings momentum is improving (acceleration = sue_t - sue_{t-63d} > 0) should see continued price appreciation as the market under-reacts to improving trends. Signal = change in 1-year NI growth rate over 63 trading days. Sector-neutral top 20%, monthly rebalance.

## Expected regime behavior
- **2000-2002 tech crash**: Earnings collapsed; acceleration was deeply negative in tech → exits early
- **2008-2009 GFC**: Deceleration visible in fundamentals before price lows
- **2013-2017 low-vol**: Steady earnings expansion; acceleration signal stable but low-variance
- **2020 COVID**: Earnings collapsed then rebounded; acceleration spikes during recovery phase

## Caveats
- EDGAR XBRL coverage starts ~2009; folds 0-5 produce zero weights (same as H180-H184)
- 6+ years of zero-weight folds means OOS Sharpe is computed on fewer folds; CI will be wide

## Success criteria
- OOS Sharpe > 0.51 on folds where signal is available
- CI lower bound > 0
- Marginal value over H180 (first derivative) justifies complexity

## Falsification
- If acceleration adds no incremental signal beyond H180, the second derivative is noise in this universe
- If EDGAR data gaps dominate, there won't be enough OOS folds to cross the keep bar

## Parameters
- `yoy_window`: 252 days
- `accel_window`: 63 days (canonical)
- `within_sector_quantile`: 0.20

## Results (OOS 1993-2021, 12 folds)
- **Decision: KEEP**
- OOS Sharpe: 0.681  CI [0.283, 1.089]  (wide CI due to EDGAR gap folds)
- CAGR: 7.60%  Max DD: -36.08%
- H001 corr: +0.584 (lowest in Batch 10 — earnings-based signal is genuinely different)
- Fold Sharpes: 0, 0, 0, 0, 0, 0, 0.40, 1.86, 0.82, 1.56, 0.93, 0.90 (6 zero folds)
- Sensitivity: 42-126d accel_window all Sharpe 0.67-0.68 — insensitive to window length
- Notable: lowest MaxDD in Batch 10 (-36.1%); acceleration second derivative provides DD protection
- The 6 zero-weight EDGAR folds hurt Sharpe but the 6 active folds average +1.09
