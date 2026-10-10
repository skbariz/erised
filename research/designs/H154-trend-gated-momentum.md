# H154: TrendGatedMomentum — H150 Gated by S&P 500 Trend Signal

## Hypothesis

Momentum crashes occur during sharp market reversals — precisely the regimes where the S&P 500 index is in a downtrend (below its 200-day MA). Scaling the H150 momentum portfolio's exposure by the trend state of the S&P 500 equal-weight index should reduce crash exposure: when the market is in a clear downtrend (index below 200-day MA), go flat; in a weak trend (between 100 and 200-day MA), hold 50%; in a clear uptrend (above 200-day MA), hold 100%. This is a pure S&P 500 strategy — the trend signal is derived from the equal-weight average of all S&P 500 members' returns, not an external ETF.

**Signal:** Compute the equal-weight index of all S&P 500 members' prices. Gate = 1.0 if index > 200dMA, 0.5 if 100dMA < index ≤ 200dMA, 0.0 if index ≤ 100dMA. Weights = H150_weights × gate.

## Expected Regime Behavior

- **2000-2002 tech crash:** Gate goes to 0 as index falls through 200dMA in 2000. Dramatic reduction in momentum exposure prevents the worst losses. Expected: significantly better than H150 in folds 0-1.
- **2008-2009 GFC:** Gate fires early (index below 200dMA by October 2008). Mostly flat through the worst months. Misses some of the March 2009 recovery (gate slow to re-open). Net: much better Max DD than H150.
- **2013-2017 low-vol bull:** Gate is permanently open (index consistently above 200dMA in these years). H154 ≈ H150 in this regime.
- **2020 COVID:** Gate briefly fires (March 2020 crash). Prevents full exposure during the drawdown. Re-opens by May 2020 for the recovery. Net: improved vs H150 for this fold.
- **2022 bear:** Gate fires progressively as index falls through MAs. Reduces exposure through the worst of the 2022 drawdown. However, momentum itself was directionally correct in 2022 (energy up, tech down) — the gate may reduce some beneficial energy-long exposure.

## Success Criteria

- OOS Sharpe > H150 Sharpe (trend gate should improve risk-adjusted returns by reducing crash exposure)
- Max DD materially better than H150 (the primary expected benefit)
- 95% CI lower bound > 0
- H001 corr within +0.05 of H150 (the gate shouldn't dramatically change the equity character)

## Falsification

- Sharpe ≤ H150 − 0.05 AND Max DD no better than H150 → the trend gate adds costs (missed recovery rallies) without reducing crashes enough; the momentum crashes happen faster than the gate can fire
- The gate is never active → the S&P 500 was above both MAs for most of the test period; the gate provides no practical risk reduction

## Implementation

- Class: `TrendGatedMomentum(CrossSectionalStrategy)` in `src/erised/strategies/cross_sectional.py`
- Params: `long_days=252`, `skip_days=21`, `top_quantile=0.20`, `ma_long=200`, `ma_short=100`
- Trend signal: equal-weight index of all S&P 500 members in `prices_panel` (all-internal, no external SPY data needed)
- Gate: full (1.0) above ma_long, half (0.5) between ma_short and ma_long, zero otherwise
- All signals lagged by 1 day (shift(1)) to avoid lookahead
- Runner: `scripts/run_h154.py`

## Results

*(to be filled after backtest)*
