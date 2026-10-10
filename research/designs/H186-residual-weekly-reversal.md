# H186: ResidualWeeklyReversal

## Hypothesis

**Signal:** Beta-adjusted short-term reversal. Strip out market beta before computing the 5-day reversal signal: `residual_5d = ret_5d - beta × spy_5d`. Long stocks with the most negative idiosyncratic 5-day return (biggest idiosyncratic losers).

**Why it should work:** H185 uses raw 5-day returns. But a large part of any stock's 5-day return is explained by the market's 5-day move. A stock that fell 5% when SPY fell 4% has only -1% idiosyncratic return. Residual reversal focuses on stocks that fell due to stock-specific reasons (news overreaction, forced selling, index rebalancing flows) rather than market co-movement. These idiosyncratic reversals are theoretically purer and more exploitable.

**This tests H185's robustness:** If H185 passes, is the alpha from raw reversal or beta-adjusted reversal? H186 answers whether stripping the market factor improves the signal.

**Expected regime behavior:**
- **2000-2002 tech crash:** Beta stripping removes the market decline component, revealing stocks that fell even more than the market. Idiosyncratic reversals here are more likely to be catching-falling-knife situations. Possibly worse than H185.
- **2008-2009 GFC:** Same concern. High-beta financials would have large idiosyncratic residuals in absolute terms.
- **2020 COVID:** Individual company overreactions should revert; market-level crash less relevant.
- **Low-vol 2013-2017:** Idiosyncratic reversals dominate. Best environment for this variant.

**Cost consideration:** Same as H185 — weekly rebalance, ~130 bps/yr annualized cost.

## Success criteria

- OOS Sharpe > H185 Sharpe (must beat raw reversal to justify complexity)
- 95% CI lower bound > 0

## Falsification

If H186 Sharpe ≤ H185 Sharpe, beta-stripping adds no value over raw reversal within the S&P 500. The market component of short-term returns either doesn't revert or the noise from beta estimation offsets the signal improvement.

## Implementation

- `ResidualWeeklyReversal` in `src/erised/strategies/cross_sectional.py`
- Beta from rolling 252-day covariance; residual_5d = rets_5d - beta × spy_5d
- Sensitivity sweep: beta_window ∈ {126, 252, 504} days

## OOS Results

**Decision: KEEP**

| Metric | Value |
|--------|-------|
| OOS Sharpe | 1.095 |
| 95% CI | [0.696, 1.513] |
| CAGR | 24.20% |
| Max DD | -51.71% |
| H001 corr | +0.613 |

Beta-stripping adds negligible value over H185 (1.095 vs 1.103). The market beta component of 5-day returns contributes slightly to the reversal signal. Sensitivity: shorter beta_window marginally better (126d: 1.106 > 252d: 1.095 > 504d: 1.071). H185 preferred due to simplicity at equivalent performance.
