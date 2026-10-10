# H644 — ShrinkageRobustChampion

**Signal:** The H622 VIX-momentum asymmetric mechanism, but with the portfolio weights **ridge-shrunk
toward the MDR prior** (objective = −lev-Sharpe + μ·‖w − w_mdr‖², μ swept). Reports the smallest μ that
keeps Sharpe near H622 while diversifying the weights.

**Why it should work:** concentrated SLSQP weights can overfit the OOS realization. Shrinkage toward the
max-diversification prior trades a sliver of in-sample Sharpe for more robust, less concentrated weights —
a standard regularization that should hold up better out of sample.

**Success:** Sharpe within 0.10 of H622 (2.642) with visibly more diversified weights (lower max weight).
**Falsify:** any μ>0 costs >0.10 Sharpe → the concentration was earned, not overfit. **Result:** appended after run.
