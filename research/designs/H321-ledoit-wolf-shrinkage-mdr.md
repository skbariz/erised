# H321: LedoitWolfShrinkageMDR -- MDR with shrinkage covariance over H001-H100 strategies

## Hypothesis
Sample covariance matrices estimated from short OOS windows are noisy -- the 30-year OOS period yields roughly 7,500 daily observations, but 35 strategies means the sample covariance has ~630 free parameters. Ledoit-Wolf (2004) analytical shrinkage reduces estimation error by pulling the sample covariance toward a structured target, particularly helping when N (strategies) is not negligible relative to T (observations). If sample covariance noise is causing H095 MDR to over-concentrate on low-vol strategies at the expense of genuine diversification, Ledoit-Wolf shrinkage should produce more stable weights and potentially higher Sharpe or lower MaxDD. The MDR objective (maximize sum(w*sigma_i) / sqrt(w^T @ Sigma @ w)) is identical to H095 -- only the covariance estimate changes. This directly tests whether covariance estimation quality is the binding constraint for H095.

**Expected behavior:**
- 2000-2002 tech crash: similar to H095 -- H043/H078 calendar heavy weighting means strategy is mostly out of equities; shrinkage should not alter this significantly since calendar strategies have near-zero covariance with trend strategies regardless of estimator
- 2008-2009 GFC: Ledoit-Wolf should produce slightly different weight allocation vs H095 if sample covariance was over-concentrating on crisis-period patterns; expect similar or slightly better drawdown behavior
- 2020 COVID: no structural difference expected -- shock is short enough that both estimators use the same training data
- 2013-2017 low vol: if shrinkage slightly redistributes weight away from the extreme H043/H078 concentration (H095 gives 24%+20% to those two), then Sharpe in quiet periods may be more stable

**Success criteria:** OOS Sharpe >= 1.10 (matching or beating H095 1.133); MaxDD < -10%; CI lower > 0; different weight distribution from H095 (top weight difference > 3%).
**Falsification:** Sharpe < H095 (1.133) by more than 0.05 or weight distribution is nearly identical to H095 (shrinkage adds nothing at this T/N ratio). If Ledoit-Wolf and sample covariance give identical weights, sample covariance is not the binding constraint.

## Results
(to be filled after running)
