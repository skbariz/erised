# H151: MomentumMarketNeutral — 12-1m Cross-Sectional Momentum, Long/Short

## Hypothesis

Long the top quintile of S&P 500 stocks by 12-1 month return, short the bottom quintile, with 0.5 gross exposure on each leg (total gross = 1.0, net ≈ 0). The long/short construction cancels most SPY beta by design, producing a near-zero market correlation. Unlike H001-H100 market-level L/S strategies (which all failed because they shorted the entire index), cross-sectional L/S shorts individual underperforming stocks within equities — the expected return differential between top and bottom quintile (~4-6% per year before costs) provides a real compensating return that pure index shorting does not have. Borrow cost of 50 bps/year on the short leg is modelled as a daily drag.

**Signal:** same as H150. `mom = price(t-21) / price(t-252) - 1`. Long top quintile (+0.5 gross), short bottom quintile (-0.5 gross). Equal-weighted within each leg.

## Expected Regime Behavior

- **2000-2002 tech crash:** Long leg is in momentum winners (tech leaders at top of bubble). Short leg is in loser sectors. As bubble bursts, both legs reverse — classic momentum crash. Expect severe fold 0-1 losses.
- **2008-2009 GFC:** Long leg held financials into the crash, then missed the recovery. Short leg (prior losers) rallied 100%+ in March 2009. This is the worst regime for momentum L/S.
- **2013-2017 low-vol bull:** Steady trend environment — momentum L/S earns the cross-sectional spread. Long leg outperforms, short leg underperforms. Expect best fold performance here.
- **2020 COVID:** Momentum reversal in March-April 2020, then recovery. Similar pattern to 2009 but shorter duration.
- **2022 bear:** Energy (prior losers becoming winners) would have been in short leg of 2021 momentum portfolio → short leg hurt. Long leg in tech → hurt by rate rise. Potential double-whammy.

## Success Criteria

- Post-borrow-cost OOS Sharpe > 0.30 (lower bar than long-only; the L/S construct sacrifices some Sharpe for near-zero market correlation)
- 95% CI lower bound > 0
- H001 corr < +0.30 (the key value of L/S is low market correlation)
- Not required to beat SPY Sharpe (0.51) due to near-zero market exposure — it is ensemble-valuable at any positive Sharpe

## Falsification

- Post-cost Sharpe < 0 or CI clips zero → transaction costs + borrow eliminate the cross-sectional spread in large-cap names
- H001 corr > +0.50 → insufficient market neutralization; the long leg dominates

## Implementation

- Class: `MomentumMarketNeutral(CrossSectionalStrategy)` in `src/quant/strategies/cross_sectional.py`
- Params: `long_days=252`, `skip_days=21`, `top_quantile=0.20`, `bottom_quantile=0.20`
- 0.5 gross on each leg; net ≈ 0 by construction (not beta-adjusted; pure equal-dollar neutralization)
- Borrow cost applied in runner as daily drag: `0.5 * 0.0050 / 252` per day
- Engine: `run_panel_backtest()` with 2.5 bps/turn
- Runner: `scripts/run_h151.py`

## Results

*(to be filled after backtest)*
