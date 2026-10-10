# H162: CompositeValueMomentum — Momentum + Long-Term Reversal Blend

## Signal
Composite z-score: `0.5 × z(12-1m momentum) + 0.5 × z(−36m return)`.

The negative sign on 36-month return means stocks that are long-term losers score positively on the reversal component. Long top quintile of the composite score. Equal-weight, monthly rebalance.

**Why it should work:** Asness, Moskowitz & Pedersen (2013) showed value and momentum have negative contemporaneous correlation across asset classes, making their combination more Sharpe-efficient than either alone. For individual stocks, combining intermediate momentum (buy recent winners) with long-term reversal (avoid recent multi-year winners who are mean-reverting) should remove a category of false momentum signals — stocks that are 12-month winners but 3-year losers are legitimate momentum; stocks that are 12-month winners AND 3-year winners are more likely to revert. The composite ranks stocks by both quality of trend and reversal valuation.

## Expected regime behavior
- **Momentum crashes (2001, 2009)**: The reversal component partially hedges momentum crashes — when momentum leaders correct, they were often also long-term winners (negative reversal score), so the composite avoided them.
- **Mean reversion recoveries**: When cheap stocks bounce, the composite ranks them high if they've also started showing positive intermediate momentum.
- The composite should be more stable across regimes than pure momentum.

## Success criteria
- OOS Sharpe > 0.60 (higher bar since it's a combination — should be better than H150 at 0.98)
- 95% CI lower bound > 0
- H001 correlation < H150's correlation (0.71) — blending in reversal should reduce market-momentum loading

## Falsification
- If Sharpe < H150 (0.98), the reversal component adds noise rather than signal at large-cap scale
- If H001 corr is similar to H150 (~0.70), the reversal component has no decorrelating effect

## Implementation notes
- Two-component z-score blend computed in cross-section each period
- Both components cross-sectionally z-scored before blending (normalize units)
- Momentum: `long_days=252, skip_days=21`
- Reversal: `reversal_days=756` (36 months), `reversal_skip=21`
