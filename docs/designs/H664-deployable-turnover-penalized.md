# H664 — DeployableTurnoverPenalized664

**Signal:** The deployable Sharpe portfolio (H650 construction, fixed at the frontier-champion config
t15w20), but the weights are fit to maximize **leveraged Sharpe − λ·‖w − w_mdr‖₁** — an L1 penalty pulling
the allocation toward the diversified **max-diversification (MDR) warm-start anchor** — on the 1-day-lagged
signal. Sweep λ ∈ {0, 0.05, 0.1, 0.2}; finalize on the **largest λ that holds within 0.05 Sharpe of the
λ=0 optimum**. Report the L1 distance from the anchor (a concentration/turnover proxy) at each λ.

**Why it should work / what it tests:** the unconstrained Sharpe fit (H650) concentrates ~85% on five
sleeves (H473, H076, H065, H265). An L1 pull toward the MDR anchor is the classic sparsity/stability
regularizer — it disciplines the optimizer away from corner solutions that an unconstrained fit reaches by
over-trusting in-sample covariances, at a controlled Sharpe cost. Distinct from the L2-toward-equal-weight
ridge (H654): L1 (not L2) and anchored on MDR (not equal weight), so it shrinks toward a *diversified-by-
risk* point rather than the naive 1/n.

**Expected behavior:** small λ should barely move Sharpe while measurably cutting the L1 distance from the
anchor (more diversified, lower turnover); large λ eventually collapses the allocation onto the MDR anchor
and costs Sharpe. The finalize rule keeps the most-diversified allocation that is still within 0.05 Sharpe.

**Success:** a λ > 0 holding within 0.05 Sharpe of the λ=0 optimum while reducing the anchor distance — i.e.
diversification is nearly free here. **Falsify:** any λ > 0 costs > 0.05 Sharpe (the concentrated fit is
load-bearing) → finalize collapses to λ=0 and the penalty buys nothing.

**Result:** appended after run.
