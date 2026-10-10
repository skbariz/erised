# H666 — DeployableHRP666

**Signal:** A **return-blind** deployable baseline that allocates the 55-sleeve pool by **Hierarchical Risk
Parity** (López de Prado 2016): build the correlation-distance matrix `d = sqrt(0.5·(1−ρ))`, single-linkage
cluster it, **quasi-diagonalize** (seriate) the covariance, then split weight by **recursive bisection**,
allocating inversely to each sub-cluster's inverse-variance-portfolio variance. The same 1-day-lagged
interior-grid leverage as H650 is then applied (threshold {15,16,17} × window {20}, plain + asymmetric).

**Why it should work / what it tests:** ERC (H659) and inverse-vol (H653) treat the 55 sleeves as a flat
list. But the pool has obvious block structure — trend sleeves, calendar sleeves, the L/S reversal sleeves
(H471/H472/H473), fundamental sleeves — that are correlated within block and near-orthogonal across blocks.
HRP respects that hierarchy: it never lets one crowded block dominate, and it allocates across blocks before
within, which is exactly where the diversification lives. It is the cluster-aware sibling of ERC and needs no
matrix inversion, so it is more stable than min-variance on a near-singular 55×55 covariance.

**Expected behavior:** HRP should land at or above ERC (H659, 1.414) — clustering should help where ERC's
flat risk-parity over-weights members of a tight block — and well below the return-optimized H650 (1.979),
since it ignores expected returns entirely. Weights should concentrate across blocks, not on single sleeves.

**Success:** deployable Sharpe ≥ the ERC baseline H659 and clearing the keep bar, with a valid simplex weight
vector and a sensible quasi-diagonal ordering. **Falsify:** HRP lands below ERC, or fails the keep bar →
cluster structure adds nothing over flat risk parity on this pool.

**Result (KEEP).** Config t15w20. Sharpe **1.223** [0.83, 1.61], Calmar 0.813, Sortino 1.283, CAGR 6.02%,
MaxDD −7.40%, H001 corr +0.673, folds 0.56–2.29. Single-linkage HRP lands **below** flat ERC (H659, 1.414) by
−0.191 and even slightly below naive inverse-vol (H653, 1.303) by −0.080. The recursive-bisection split is
more defensive (the batch's lowest CAGR, 6.02%, and a tight −7.40% MaxDD) but without the Sharpe to match ERC:
on this 55-sleeve pool the cluster hierarchy adds nothing over flat risk-contribution parity — **the falsify
clause fires.** Kept as an honest return-blind baseline (CI clears zero), not a frontier strategy.
