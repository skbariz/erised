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
