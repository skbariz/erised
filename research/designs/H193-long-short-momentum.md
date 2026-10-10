# H193: LongShortMomentum

## Hypothesis

**Signal:** Market-neutral long-short momentum. Long top-20% momentum stocks, short bottom-20% momentum stocks, sector-neutral. Monthly rebalance. Extends H150 (MomentumLongOnly) by adding a short leg.

**Why it should work:** Academic momentum (Jegadeesh-Titman 1993) is originally defined as long-short. Long-only momentum (H150) captures the long leg only. The short leg (shorting prior losers) should add alpha: losers continue to underperform in the subsequent month. Market-neutral construction eliminates beta exposure, so alpha comes purely from the cross-sectional momentum spread.

**Key question:** Does the short leg add value in practice within S&P 500?
- Short leg advantages: adds alpha from continued underperformance of losers, reduces market exposure
- Short leg disadvantages: doubles turnover cost, short sellers may face short squeezes, S&P 500 losers are often rebalanced out (reducing the loser universe quality)

**Expected regime behavior:**
- **2000-2002 tech crash:** Strong momentum reversal in 2001-2002. Short leg also loses (short the recent losers who recover). Double-jeopardy in momentum crash.
- **2008-2009 GFC:** Momentum crash of 2009 is famous — strategy would suffer on both legs simultaneously.
- **2020 COVID:** Sharp momentum reversal in May 2020. Same risk.
- **Low-vol 2013-2017:** Pure momentum environment. Both legs work well.

**H001 correlation:** Near zero by construction (market-neutral).

## Success criteria

- OOS Sharpe > 0.51
- 95% CI lower bound > 0
- H001 corr < 0.20 (market-neutral by design)
- Compare to H150 (0.981) — if Sharpe > H150, short leg adds value

## Falsification

If Sharpe < H150 Sharpe (0.981): the short leg either adds no alpha (S&P 500 losers don't continue to underperform) or the extra cost (doubled turnover) eliminates it. If Sharpe < 0.51 and CI clips zero: L/S momentum is outright destructive in S&P 500.

## Implementation

- `LongShortMomentum` in `src/erised/strategies/cross_sectional.py`
- Long = `_sector_neutral_weights(mom, ...)`, short = `_sector_neutral_weights(-mom, ...)`
- Combined: `(long_w * 0.5) - (short_w * 0.5)`, normalized to abs_sum = 1.0
- Sensitivity sweep: mom_window ∈ {126, 189, 252} days

## OOS Results

**Decision: kill**

| Metric | Value |
|--------|-------|
| OOS Sharpe | -0.157 |
| 95% CI | [-0.546, 0.245] |
| CAGR | -1.22% |
| Max DD | -41.27% |
| H001 corr | +0.116 |

Negative Sharpe. L/S momentum is destroyed by momentum crashes: fold 5 (GFC 2008-2010) is -1.625. The short leg (short recent losers) amplifies losses when momentum crashes — both legs go against the strategy simultaneously. This confirms the H001-H100 finding: long-short with equity factors adds crash risk rather than hedge value. Long-only H150 (Sharpe 0.981) is far superior.
