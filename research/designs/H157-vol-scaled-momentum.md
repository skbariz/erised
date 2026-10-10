# H157: VolScaledMomentum — Vol-Target Portfolio Scaling (Barroso & Santa-Clara 2015)

## Hypothesis

Momentum crashes occur in high-volatility regimes. Barroso & Santa-Clara (2015) showed that the momentum portfolio's own realized volatility is time-varying and negatively predicts returns: when the momentum portfolio's volatility is high, reduce exposure; when it's low, run at full (or levered) exposure. This is exactly H086 (LowVolTargetPortfolio, Sharpe 1.109) applied to the cross-sectional momentum portfolio instead of H021.

**Signal:** Build H150 (12-1 momentum, top quintile). Compute trailing 21-day realized volatility of the portfolio's return series. Exposure multiplier = `min(1.5, max(0.2, 12% / annualized_portfolio_vol))`. Scale all weights daily; stock selection updates monthly.

## Expected Regime Behavior

- **2000–2002 tech crash:** As the momentum crash hits in 2000–2001, portfolio vol spikes → multiplier drops below 0.5 → exposure cut automatically. Expected: dramatically better fold 1 vs H150.
- **2008–2009 GFC:** Portfolio vol surges in September–November 2008 → near-zero exposure through the worst months. Misses some of the March 2009 recovery (vol slow to fall). Net: much better Max DD.
- **2013–2017 low-vol bull:** Portfolio vol is consistently low → multiplier near or above 1.0 → H157 ≈ H150 or slightly leveraged in this regime.
- **2020 COVID:** Vol spike in March 2020 → exposure cut. Recovers by May 2020. Net improvement vs H150 for this period.
- **2022 bear:** Momentum vol elevated through 2022 → partial exposure reduction. Momentum was directionally correct in 2022 (energy up) but H157 still cuts some of the beneficial exposure.

## Success Criteria

- OOS Sharpe > H150 + 0.05 (vol scaling should improve Sharpe, as H086 did for H021)
- Max DD materially better than H150 (the primary benefit)
- CI lower bound > 0
- Sensitivity: robust to vol_target ∈ {8%, 12%, 16%} and vol_window ∈ {10, 21, 42}

## Falsification

- Sharpe ≤ H150 AND Max DD no better → the momentum portfolio's vol spikes are not forecastable; the vol scaling cuts exposure at the wrong times (before recoveries rather than before crashes)
- Multiplier always near 1.0 → momentum portfolio vol is too stable; the scaling adds no exposure management

## Implementation

- Class: `VolScaledMomentum(CrossSectionalStrategy)` in `src/erised/strategies/cross_sectional.py`
- Params: `long_days=252`, `skip_days=21`, `top_quantile=0.20`, `vol_target=0.12`, `vol_window=21`, `vol_min=0.2`, `vol_max=1.5`, `monthly_rebalance=True`
- Mechanism: stock selection via H150 (monthly), exposure scaling via vol-target (daily)
- Uses UNSCALED portfolio returns to estimate vol — no circular dependency
- Runner: `scripts/run_h157.py`

## Results

*(to be filled after backtest)*
