# H560: ConcentratedPoolMDR

## Signal
Apply H558's winning leveraged-Calmar SLSQP objective to reduced pools (top 20 and top 30
strategies by base Sharpe) selected from the full 55-strategy pool.

## Hypothesis
H516 (Set7 Batch7) found that a focused 20-strategy pool cracked the MDR ceiling vs the
full pool — Sharpe 1.577. In the Set8 context: when the SLSQP optimizer has fewer, higher-quality
strategies to choose from, it may find a better optimum by avoiding the noise and correlation
of mediocre strategies. H558's full-pool leveraged-Calmar produced Cal_lev 1.802, CAGR 11.89%.

## Pool selection
- Run all 55 strategies to get their base Sharpe
- Top 20: strategies ranked 1-20 by base Sharpe
- Top 30: strategies ranked 1-30 by base Sharpe
- Apply leveraged-Calmar objective (H558's winner) to each pool size
- Also run MDR (H595-style) on each pool for comparison

## Expected behavior
- Smaller pool: optimizer concentrates on H473/H157/H065/H076 (the heavy hitters), may find
  higher Cal_lev by avoiding correlation with weaker strategies
- Larger pool: more diversification options, but more optimization noise
- 2008-2009 GFC: concentrated pool may have worse drawdown if it drops low-corr strategies
- 2013-2017 low-vol: concentrated quality pool should shine

## Success criteria
- Cal_lev > 1.802 (H558 baseline) for at least one pool size
- OR Sh_lev > 1.790 (H556 baseline)
- CI lower bound > 0, all 12 folds positive

## Falsification
If both top-20 and top-30 underperform H558, larger diverse pools are better for leveraged
SLSQP optimization -- H516's finding does not generalize to this context.
