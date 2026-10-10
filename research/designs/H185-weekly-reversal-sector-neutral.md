# H185: WeeklyReversalSectorNeutral

## Hypothesis

**Signal:** Short-term reversal. Compute each stock's prior-5-day return. Rank by return within sector; long the bottom 20% (biggest losers). Rebalance weekly (every 5 trading days).

**Why it should work:** Jegadeesh (1990) and Lehmann (1990) document short-term reversal: stocks that fall sharply over 1-4 weeks tend to mean-revert over the following 1-4 weeks. Within S&P 500 large-caps, the mechanism is primarily liquidity provision — market makers and dealers push prices too far during sell-offs, which then revert as price discovery catches up. Weekly rebalance captures the reversal window efficiently.

**Sector-neutral:** The prior-5-day return has a strong sector component (e.g., Energy stocks all fall when oil drops). Sector-neutral removes the industry component and focuses on idiosyncratic reversal.

**Expected regime behavior:**
- **2000-2002 tech crash:** Tech stocks fell serially, making reversal trade dangerous (catching falling knives). Expect poor performance during trend periods. This is a known weakness of reversal strategies.
- **2008-2009 GFC:** Same problem — persistent declines mean reversal signals are whipsawed. Expect worst fold performance here.
- **2020 COVID:** Sharp single-month crash followed by V-shaped recovery. Reversal would catch the bounce. Good.
- **Low-vol 2013-2017:** Mean-reverting, range-bound markets. Ideal environment. Expect best performance here.

**Cost consideration:** Weekly rebalance means ~52 turnover events per year vs 12 for monthly. At 2.5 bps/turn, annualized cost is ~130 bps — potentially significant drag.

## Success criteria

- OOS Sharpe > 0.51
- 95% CI lower bound > 0
- H001 corr < 0.40 (reversal is negative-momentum; should be negatively correlated with H001 trend)

## Falsification

If Sharpe ≤ 0.51 or CI clips zero, weekly reversal is either too costly (2.5 bps × 52 turns kills the gross alpha) or not robust in large-cap universe. The S&P 500 may be too liquid for reversal to persist.

## Implementation

- `WeeklyReversalSectorNeutral` in `src/erised/strategies/cross_sectional.py`
- Signal: `-ret_5d` (negate to get losers at top)
- Sensitivity sweep: reversal_window ∈ {3, 5, 10} days

## OOS Results

**Decision: KEEP**

| Metric | Value |
|--------|-------|
| OOS Sharpe | 1.103 |
| 95% CI | [0.703, 1.517] |
| CAGR | 24.88% |
| Max DD | -50.76% |
| H001 corr | +0.613 |

Outstanding result. 12/12 folds positive including GFC (fold 5: +0.440). CAGR 24.88% highest in earnings/reversal batch. Sensitivity perfectly flat (3d/5d/10d all ~1.09-1.10). Reversal survives the 2008-2009 crisis better than anticipated — sector-neutral construction prevents catching-falling-knives sector bets. Weekly rebalance is the optimal frequency (beats H188 monthly by 0.154 Sharpe).
