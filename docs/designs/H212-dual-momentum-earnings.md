# H212: DualMomentumEarnings

**Signal:** AND condition -- top 30% of both price momentum and NI YoY growth
**Date:** 2026-05-25

## Hypothesis

Higher-conviction positions by requiring both signals: 12-1m price momentum AND YoY NI growth both in top 30% cross-sectionally. Market and earnings confirmation together. Expected fewer positions but higher quality.

## Results

**Decision: KEEP**

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.587 |
| 95% CI | [0.184, 0.991] |
| CAGR | 7.14% |
| Max DD | −36.26% |
| H001 corr | −0.009 |

Sensitivity (top_quantile): 20% 0.683, 30% 0.587, 40% 0.665. 20% threshold performs best -- tighter filter is actually better.

AND condition reduces Sharpe vs H150 (0.981 pure momentum). The dual requirement filters out many valid momentum opportunities where earnings lag price. Best contribution is MaxDD (−36.26%) -- lowest in batch for an active-folds-only strategy. H001 corr −0.009 near zero. The filtering benefit is captured in MaxDD, not Sharpe.
