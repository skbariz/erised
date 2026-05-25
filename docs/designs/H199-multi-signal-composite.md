# H199: MultiSignalComposite

## Hypothesis
Three orthogonal signals tested individually — price momentum (H150), short-term reversal (H185), and earnings surprise (H180) — each carry independent alpha. Combining them via cross-sectional z-score blending should reduce the noise of any single signal while preserving their combined predictive content. Signal = (z_mom + z_rev + z_sue) / n_available. Each signal z-scored cross-sectionally before blending; NaN signals (earnings gaps) degrade gracefully to the average of available signals. Sector-neutral top 20%, monthly rebalance.

## Expected regime behavior
- **2000-2002 tech crash**: Momentum turns negative (flips reversal direction); earnings collapse; composite adapts
- **2008-2009 GFC**: Short-term reversal (mean reversion) provides some hedge during high-volatility periods
- **2013-2017 low-vol**: Momentum dominates in stable uptrend; composite tracks H150 closely
- **2020 COVID**: Earnings signal available post-2009; reversal provides recovery signal after crash

## Key design choice
Reversal signal uses -pct_change(5) so that recent underperformers score positively (same direction as momentum/earnings in the composite). The momentum and earnings signals reinforce each other; reversal provides counter-cyclical balance.

## Success criteria
- OOS Sharpe > 0.51
- CI lower bound > 0
- Sharpe improvement over H150 alone, confirming signal blending adds value
- H001 corr < 0.80

## Falsification
- If composite Sharpe < H150, the reversal or earnings signals are adding noise, not signal
- If all three signals are positively correlated, blending provides false diversification

## Parameters
- `mom_window`: 252 days, `skip_window`: 21 days
- `reversal_window`: 5 days (canonical)
- `yoy_window`: 252 days
- `within_sector_quantile`: 0.20
