# H273: PriceBreakoutMomentum — 63-Day High Breakout Signal

## Hypothesis

George and Hwang (2004) show that 52-week high proximity is one of the strongest momentum
predictors. H195 tested the 252-day version. H273 tests a shorter 63-day window, which captures
more recent institutional sponsorship and is faster to signal.

Signal: close > rolling(63d).max().shift(1) (prior day's 63-day max, shift to avoid lookahead).
Equal-weight stocks in breakout state. Sector-neutral top-quintile, monthly rebalance.

Note: The original plan combined breakout with volume confirmation (volume gate). Volume data
unavailable in individual stock parquets (adj_close only). Testing pure price breakout.

Distinction from H195 (52-week high momentum): H195 uses 252-day window. H273 uses 63 days —
captures more recent, tighter breakouts. Expected higher turnover but faster signal response.

Expected regime behavior:
- **2000–2002**: Strong through 1999; collapses in 2000–2001 as breakouts fail.
- **2008–2009**: Weak — downtrend means very few 63-day high breakouts.
- **2013–2017**: Strong — persistent uptrend generates steady breakout signals.
- **2020**: Sharp drawdown in March; recovers quickly in April–May breakouts.

Success criteria: OOS Sharpe ≥ 0.78, CI lower > 0, H195 corr < 0.80.
Falsification: H195 correlation > 0.85 — 63d breakout is not distinct from 252d breakout.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
