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
