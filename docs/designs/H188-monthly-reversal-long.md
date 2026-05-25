# H188: MonthlyReversalLong

## Hypothesis

**Signal:** Prior-21-day return reversal. Long bottom-20% sector-neutral losers. Rebalance monthly (every 21 trading days).

**Why it should work:** This is the same reversal concept as H185 but at monthly frequency. Monthly reversal is documented by Jegadeesh (1990) as the "1-month return reversal" effect. Monthly rebalance reduces turnover by 5× relative to H185 (12 vs 52 events/year), cutting cost drag from ~130 bps to ~26 bps/yr. The question is whether the monthly signal (21-day prior return) is strong enough to survive after cost reduction and the longer holding period.

**H185 vs H188 comparison:** This is a direct cost-frequency tradeoff test:
- H185: Weekly rebalance on 5-day signal. High frequency, high cost, fresh signal.
- H188: Monthly rebalance on 21-day signal. Lower frequency, lower cost, signal averages over a month.

If H188 Sharpe ≈ H185 Sharpe, reversal alpha is robust to rebalance frequency. If H188 > H185, costs are killing H185. If H188 < H185, weekly rebalance is worth the extra cost.

**Expected regime behavior:**
- Same directional bets as H185 but held longer. More exposure to "catching falling knives" in trends.
- Monthly rebalance reduces whipsaw but increases holding period through adverse trends.
- **2013-2017 low-vol:** Monthly reversal in flat markets — good. Monthly hold is fine.
- **2008-2009 GFC:** Monthly losers become deeper losers. Monthly hold makes it worse.

## Success criteria

- OOS Sharpe > 0.51
- 95% CI lower bound > 0
- If Sharpe ≥ H185 Sharpe, prefer H188 (lower cost, same return = better risk-adjusted alpha)

## Falsification

If Sharpe < H185 Sharpe AND Sharpe < 0.51, the 21-day signal is weaker than the 5-day signal and neither frequency saves it. Kill the family.

## Implementation

- `MonthlyReversalLong` in `src/quant/strategies/cross_sectional.py`
- Signal: `-ret_21d`, sector-neutral top 20%
- Monthly rebalance (uses `_monthly_rebalance()` helper)
- Sensitivity sweep: reversal_window ∈ {10, 21, 42} days

## OOS Results

*(to be filled after backtest)*
