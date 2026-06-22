# H667 — DeployableMinCorr667

**Signal:** A **return-blind** deployable baseline using a **minimum-correlation** allocation: score each
sleeve by `1 − mean off-diagonal correlation` (low- or negative-correlation sleeves score highest), tilt the
weight by that score divided by the sleeve's volatility, and normalize. The same 1-day-lagged interior-grid
leverage as H650 is then applied (threshold {15,16,17} × window {20}, plain + asymmetric).

**Why it should work / what it tests:** the program's most valuable sleeves are the *diversifiers* — the
zero-corr seasonal H078, the drawdown-triggered H098, and the dollar-neutral L/S reversal H473 — that ERC and
inverse-vol size purely on variance, ignoring *how a sleeve correlates with the rest of the book*. The
minimum-correlation heuristic explicitly up-weights the sleeves that move independently of the crowd and
down-weights redundant, mutually-correlated sleeves. It is a cheap, inversion-free correlation tilt: a
correlation-rank cousin of ERC (H659) that pushes harder on the orthogonal sleeves than a variance solve does.

**Expected behavior:** vs ERC (H659, 1.414) it should redistribute weight from the crowded trend/parity block
toward the low-corr seasonal and L/S sleeves; Sharpe near or above ERC if that tilt is rewarded, with a
visibly lower H001 correlation than the variance-only baselines.

**Success:** deployable Sharpe ≥ ERC (H659) and clearing the keep bar, with a valid simplex weight vector that
demonstrably over-weights the known low-correlation diversifiers. **Falsify:** Sharpe below ERC, or no keep →
a correlation tilt adds nothing beyond what variance-based parity already captures on this pool.
