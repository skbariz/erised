# H152: HighWaterMomentum — 52-Week High Proximity Signal

## Hypothesis

Stocks near their 52-week high continue to outperform because investors anchor to that level and are slow to push prices above it, creating systematic underreaction to positive information. Once a stock does break above its 52-week high, continuation follows. George & Hwang (2004) showed this signal is distinct from and complementary to 12-1 month momentum (H150): a stock can have high 12-1 return but be far from its 52-week high (e.g., partially recovered from a crash), or a steady compounder can be near its 52-week high with only modest 12-1 momentum.

**Signal:** `ratio = close(t) / rolling_max(close, 252 days)`. Rank descending. Long top quintile (~100 names). Rebalance monthly.

## Expected Regime Behavior

- **2000-2002 tech crash:** Stocks near their 52-week high in 1999-2000 were tech leaders at peak bubble valuations. The high-ratio portfolio would have been loaded with bubble names — expect fold 0-1 underperformance similar to H150.
- **2008-2009 GFC:** Stocks near 52-week high in 2007 were financials and energy at their peaks — same crash risk as momentum. The 52-week high signal has heavy overlap with momentum in trending regimes.
- **2013-2017 low-vol bull:** Stable trending market with many stocks making new highs — most favorable regime. Expect strongest fold performance.
- **2020 COVID:** Rapid crash then recovery. After March 2020 bottom, the stocks near their 52-week high would be quality/healthcare names that recovered fastest — should perform well through year-end.
- **2022 bear:** Energy names would dominate the high-ratio signal (they recovered from 2020 lows to new highs) while tech declined. Better than H150 in 2022 because the signal is not anchored to 12 months of tech gains.

## Success Criteria

- OOS Sharpe > 0.55 (should be close to H150; this is an alternative momentum measure)
- 95% CI lower bound > 0
- Sharpe within ±0.10 of H150 (tests whether the anchoring signal adds genuine diversity or is just a different parameterization of the same effect)
- H001 corr < +0.80

## Falsification

- Sharpe ≤ H150 Sharpe − 0.10 → the 52-week high ratio adds no value over standard 12-1 momentum; use H150 instead
- High correlation with H150 (> 0.85) → the two signals are measuring the same thing; not worth including both

## Implementation

- Class: `HighWaterMomentum(CrossSectionalStrategy)` in `src/quant/strategies/cross_sectional.py`
- Params: `lookback=252`, `top_quantile=0.20`, `monthly_rebalance=True`
- Rolling max uses min_periods=lookback//2 to handle early periods
- Signal computed at t-1 (shift(1)) to avoid same-day lookahead
- Runner: `scripts/run_h152.py`

## Results

*(to be filled after backtest)*
