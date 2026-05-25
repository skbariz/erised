# H187: LiquidityFilteredReversal

## Hypothesis

**Signal:** Reversal restricted to illiquid stocks. Use rolling 21-day realized volatility as an Amihud illiquidity proxy (within the S&P 500, higher vol stocks tend to have wider bid-ask spreads and more erratic liquidity). Only trade the top `illiq_quantile=0.50` of stocks by volatility; within those, go long 5-day losers.

**Why it should work:** Avramov, Chordia, and Goyal (2006) show that short-term reversal profits are concentrated in illiquid stocks, where temporary price pressure creates larger deviations from fair value. Liquid large-caps revert less because arbitrage capital eliminates mispricings quickly. Within the S&P 500, the most "illiquid" (highest vol) stocks are not truly illiquid, but they represent the subset with most reversal potential.

**Proxy validity concern:** Without volume data, realized volatility is an imperfect illiquidity proxy. Within S&P 500, the correlation between vol and illiquidity is meaningful but noisy. This is an acknowledged approximation.

**Expected regime behavior:**
- High-vol stocks during **2008-2009 GFC**: Financials and cyclicals are the most "illiquid" by this metric. Strategy concentrates in worst-performing sectors during the crisis.
- **2020 COVID:** Airline/energy stocks become most volatile. Risk of catching falling knives is highest.
- **Low-vol 2013-2017:** Relatively stable high-vol stocks show genuine reversal patterns. Best environment.

**Risk:** Concentrating in the most volatile 50% of stocks amplifies reversal-in-trending-markets risk. MaxDD could be worse than H185.

## Success criteria

- OOS Sharpe > H185 Sharpe (liquidity filter should improve gross alpha enough to compensate)
- 95% CI lower bound > 0

## Falsification

If H187 Sharpe ≤ H185, the illiquidity filter either: (a) doesn't correctly identify stocks with reversal premium, or (b) the stocks identified are actually in trending downturns where reversal fails. The vol proxy is insufficient.

## Implementation

- `LiquidityFilteredReversal` in `src/quant/strategies/cross_sectional.py`
- Illiquidity proxy: `rolling_std(21d)`; top `illiq_quantile=0.50` by vol
- Sensitivity sweep: illiq_quantile ∈ {0.30, 0.50, 0.70}

## OOS Results

*(to be filled after backtest)*
