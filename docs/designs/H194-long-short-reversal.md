# H194: LongShortReversal

## Hypothesis

**Signal:** Market-neutral long-short weekly reversal. Long top-20% prior-5-day losers, short top-20% prior-5-day winners, sector-neutral. Weekly rebalance. Extends H185 (WeeklyReversalSectorNeutral) by adding a short leg.

**Why it should work:** H185 (long-only reversal) achieved Sharpe 1.103 — the best in Batch 8. Long-only captures only the reversal of losers. The short leg adds: (1) alpha from continued underperformance of recent winners (anti-continuation), (2) market-neutral construction eliminates directional market exposure. If reversal is symmetric (winners also give back gains), the short leg should add roughly equal alpha to the long leg.

**Cost consideration:** L/S doubles turnover relative to H185. H185 at ~52 turns/year costs ~130 bps. H194 at ~104 turns/year (both legs) costs ~260 bps/year. The gross alpha of the short leg must exceed ~130 additional bps to justify it.

**H001 correlation:** Should be near zero (market-neutral). If short leg works as expected, H001 corr < H185's +0.613.

**Expected regime behavior:**
- Same directional basis as H185, but with market hedge
- **2008-2009 GFC:** Long losers (which continue falling) + short winners (which also fall eventually). The hedge should help — not pure reversal failure.
- **Low-vol 2013-2017:** Both legs profitable. Best environment.

## Success criteria

- OOS Sharpe > H185 Sharpe (1.103) — short leg must add value beyond H185
- OR: OOS Sharpe > 0.51 AND H001 corr < 0.20 (accept lower Sharpe if uncorrelated — valuable for ensemble)
- 95% CI lower bound > 0

## Falsification

If Sharpe ≤ 0 or CI includes strongly negative values: the short leg reverses the strategy's alpha. Winners don't give back — they continue to outperform, and shorting them is destructive. The short leg's cost exceeds its gross alpha.

## Implementation

- `LongShortReversal` in `src/quant/strategies/cross_sectional.py`
- Long = `_sector_neutral_weights(-ret_5d, ...)`, short = `_sector_neutral_weights(ret_5d, ...)`
- Combined: dollar-neutral, abs_sum = 1.0
- Sensitivity sweep: reversal_window ∈ {3, 5, 10} days

## OOS Results

*(to be filled after backtest)*
