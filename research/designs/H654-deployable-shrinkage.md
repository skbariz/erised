# H654 — DeployableShrinkage654

**Signal:** The deployable Sharpe portfolio (H650 construction) with the optimized weights **shrunk toward
equal weight** via a ridge penalty: maximize leveraged Sharpe − λ·‖w − 1/n‖² on the 1-day-lagged signal,
sweeping λ ∈ {0, 1, 5, 20}. Deployable, lag-aware counterpart of the same-day H644 shrinkage champion.
Finalize on the largest λ that holds deployable Sharpe within 0.05 of the λ=0 optimum.

**Why it should work:** unconstrained Sharpe-optimal weights can over-concentrate on in-sample winners.
Ridge shrinkage toward equal weight regularizes the allocation, trading a little in-sample Sharpe for
out-of-sample robustness — directly addressing the Set10 overfitting concern.

**Success:** a positive λ retains deployable Sharpe within 0.05 of the unshrunk optimum with measurably
flatter weights → robustness is nearly free. **Falsify:** any shrinkage costs > 0.1 Sharpe → the weights
are already well-conditioned and concentration is earned.

**Result (KEEP).** Config t15w20, chosen **λ=1**. Deployable **Sharpe 1.945** [1.55, 2.34], Calmar 1.305,
Sortino 2.007, CAGR 10.00%, MaxDD −7.66%, H001 corr +0.447, folds 1.19–2.84. λ-sweep
{0: 1.980, 1: 1.945, 5: 1.811, 20: 1.597}: light shrinkage toward equal weight costs only **−0.035 Sharpe**
— robustness is nearly free, consistent with H644's finding that the H650 weights are earned, not overfit.
