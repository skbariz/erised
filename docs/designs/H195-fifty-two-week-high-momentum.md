# H195: FiftyTwoWeekHighMomentum

## Hypothesis
George and Hwang (2004) show that the ratio of a stock's current price to its 52-week high is a stronger momentum predictor than raw 12-1m returns. The mechanism is anchoring bias: analysts and investors use the 52-week high as a reference point and under-react when prices approach it. Stocks with price/52w-high ratios near 1.0 have strong momentum because resistance to crossing the high is eventually overcome. Signal = price / rolling_max(252d). Sector-neutral top 20%, monthly rebalance.

## Expected regime behavior
- **2000-2002 tech crash**: Stocks hitting new lows have low scores; signal exits tech early
- **2008-2009 GFC**: All-sector drawdown, scores collapse uniformly; sector-neutral should limit damage
- **2013-2017 low-vol**: Anchoring bias is persistent; steady upward regime should show strong signal
- **2020 COVID**: Sharp drop then recovery; signal may lag due to anchoring effect at pre-crash highs

## Success criteria
- OOS Sharpe > 0.51 (beat SPY benchmark)
- 95% CI lower bound > 0
- H001 correlation < 0.80 (should differ from raw trend following)

## Falsification
- If OOS Sharpe < 0.40 across all folds, anchoring bias is already arbitraged in large-caps
- If H001 corr > 0.90, this is just a repackaging of the trend signal

## Parameters
- `high_window`: 252 days (canonical)
- `within_sector_quantile`: 0.20
- `rebalance_period`: 21 days

## Results (OOS 1993-2021, 12 folds)
- **Decision: KEEP**
- OOS Sharpe: 0.878  CI [0.475, 1.286]
- CAGR: 14.43%  Max DD: -45.83%
- H001 corr: +0.692
- Fold Sharpes: 1.23, 0.57, 0.77, 1.56, 1.32, -0.25, 0.97, 2.06, 1.05, 1.65, 1.17, 0.96
- Sensitivity: high_window 126-252d all Sharpe 0.85-0.88 — stable
- Beats H150 (0.981) on MaxDD (-45.8% vs -52.1%) but lower Sharpe; anchoring adds some DD protection
