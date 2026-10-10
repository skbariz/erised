# H150: MomentumLongOnly — 12-1m Cross-Sectional Momentum, Long-Only

## Hypothesis

Among S&P 500 constituent stocks, the top quintile by 12-1 month return (price 21 trading days ago divided by price 252 trading days ago, minus 1) continues to outperform the index over the next month. This is the Jegadeesh-Titman (1993) baseline applied to the point-in-time S&P 500 universe. The portfolio holds approximately 100 stocks equal-weighted, rebalanced on the first trading day of each month. This is the baseline for the entire H150-H200 cross-sectional program — all subsequent momentum variants (H151-H159) are evaluated against this strategy.

**Signal:** `mom = price(t-21) / price(t-252) - 1`. Rank descending within members. Long top 20% (~100 stocks). Equal-weight. Monthly rebalance.

## Expected Regime Behavior

- **2000-2002 tech crash:** Momentum held tech leaders late into the decline. Expect fold 0-1 underperformance — stocks at all-time highs in 1999 momentum portfolio then crashed 80%+.
- **2008-2009 GFC:** Momentum loaded on financials in 2007-early 2008, then caught the crash. March 2009 reversal is the classic momentum crash: prior losers (financials) rallied 100%+ while prior winners lagged. Expect fold 3 losses.
- **2013-2017 low-vol bull:** Momentum thrives in steady trending markets. Expect strong folds here — tech and consumer discretionary leaders compounded without reversal.
- **2020 COVID:** Brief momentum crash (Feb-Apr 2020 when energy and financials were in top quintile), then rapid recovery. Growth stocks (prior winners) led the rebound.
- **2022 bear:** 2021 momentum portfolio was heavily tech/growth → caught in 2022 rate-driven selloff. Expect weak fold performance.

## Success Criteria

- OOS Sharpe > 0.60 (modest hurdle — this is the most documented academic factor)
- 95% CI lower bound > 0
- Post-cost Sharpe positive
- H001 corr < 0.80 (long-only equity will correlate with SPY; below 0.80 shows stock-specific alpha)

## Falsification

- Post-cost Sharpe < 0.51 → momentum has been arbitraged or is too costly at monthly rebalance in large-cap S&P 500
- All momentum variants (H151-H154) should be killed if H150 fails — the baseline signal is absent

## Implementation

- Class: `MomentumLongOnly(CrossSectionalStrategy)` in `src/erised/strategies/cross_sectional.py`
- Params: `long_days=252`, `skip_days=21`, `top_quantile=0.20`, `monthly_rebalance=True`
- Monthly rebalance: weights computed daily but only applied on month-start dates; positions held constant intra-month
- Strict point-in-time membership: only rank stocks in `membership` on signal date
- Engine: `run_panel_backtest()` with 2.5 bps/turn costs
- Runner: `scripts/run_h150.py`

## Results

*(to be filled after backtest)*
