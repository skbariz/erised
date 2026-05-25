# H189: ResidualOUMeanReversion

## Hypothesis

**Signal:** Avellaneda-Lee (2010) Ornstein-Uhlenbeck process applied to idiosyncratic returns. For each stock, compute: (1) market beta via rolling 252d regression, (2) 5-day cumulative idiosyncratic return (residual), (3) z-score of that residual against its 252-day history. Enter long when z-score < -1.5 (stock is abnormally low relative to its own OU process).

**Why it should work:** Avellaneda and Lee (2010) formalize the statistical arbitrage intuition: residual stock prices follow mean-reverting OU processes after removing systematic factor exposure. When the OU process's normalized deviation (z-score) is extreme (< -1.5), the stock is more likely to mean-revert than continue declining. This is theoretically principled unlike simple price reversal.

**Key differences from H185/H186:**
- H185: Raw price losers → reversal (no distributional assumption)
- H186: Beta-adjusted losers → reversal (removes market, but no OU formalism)
- H189: Beta-adjusted, OU z-score threshold entry (enters only when deviation is statistically extreme)

The z-score threshold acts as a filter: only enter trades with sufficient mean-reversion probability. This should reduce trade count but improve quality.

**Expected regime behavior:**
- **2000-2002 tech crash:** Tech stocks have very negative idiosyncratic z-scores throughout. Strategy would be heavily long tech. Potentially catching falling knives.
- **2008-2009 GFC:** Same risk. OU process breaks down in regime changes — the "mean" shifts down permanently.
- **2020 COVID:** V-shaped recovery. z-scores reach extreme negatives on March 23, 2020; strategy would be fully long on the recovery. Strong positive performance expected.
- **Low-vol 2013-2017:** Genuine OU mean reversion in stable regimes. Best environment.

**OU assumption validity:** Avellaneda-Lee was designed for pairs/baskets, not individual stocks. Individual stocks can have persistent idiosyncratic trends (good or bad management). The 252-day history provides local OU calibration that adjusts for trend — but not fast enough in regime changes.

## Success criteria

- OOS Sharpe > 0.51
- 95% CI lower bound > 0
- H001 corr < 0.20 (OU mean reversion should be uncorrelated with trend following)
- Fewer trades than H185 (threshold filtering reduces trade count)

## Falsification

If Sharpe ≤ 0.51 or CI clips zero, individual stock OU processes are not exploitable — either: (a) the OU assumption is too fragile for individual stocks (persistent trends violate mean-reversion), or (b) z < -1.5 threshold is too conservative (misses enough reversals) or too aggressive (includes persistent downtrends).

## Implementation

- `ResidualOUMeanReversion` in `src/quant/strategies/cross_sectional.py`
- Beta from rolling 252d; residual OU z-score computed on 5d cumulative residuals vs 252d history
- Entry: z < entry_z = -1.5; sector-neutral, weekly rebalance
- Sensitivity sweep: entry_z ∈ {-1.0, -1.5, -2.0}

## OOS Results

*(to be filled after backtest)*
