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

- `LiquidityFilteredReversal` in `src/erised/strategies/cross_sectional.py`
- Illiquidity proxy: `rolling_std(21d)`; top `illiq_quantile=0.50` by vol
- Sensitivity sweep: illiq_quantile ∈ {0.30, 0.50, 0.70}

## OOS Results

**Decision: KEEP**

| Metric | Value |
|--------|-------|
| OOS Sharpe | 1.068 |
| 95% CI | [0.666, 1.483] |
| CAGR | 25.64% |
| Max DD | -53.17% |
| H001 corr | +0.600 |

Liquidity filter does NOT improve over H185 (1.068 vs 1.103). Concentrating in high-vol names reduces Sharpe slightly. Sensitivity: illiq_quantile=70% (1.084) > 50% (1.068) > 30% (1.009) — more concentration marginally helps but MaxDD worsens. Highest CAGR in batch (25.64%) but worse Sharpe due to higher volatility. H185 preferred.
