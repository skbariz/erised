# H659 — DeployableERC659

**Signal:** A **non-optimized** deployable baseline: pool weights set by **equal-risk-contribution** (ERC,
correlation-aware risk parity over the 55-sleeve covariance) — each sleeve contributes the same share of
portfolio variance — under the same 1-day-lagged interior-grid leverage as H650. No return-fitting SLSQP:
the allocation depends only on the covariance, so it cannot be over-fit to in-sample returns. Sweep the
interior grid (threshold {15,16,17} × window {20}, plain + asymmetric); finalize on the best leverage config.

**Why it should work / what it tests:** H653 set weights by inverse-vol, which ignores correlations and so
double-counts clustered sleeves. ERC fixes that by equalizing *risk contributions* (w_i·(Σw)_i), the
textbook risk-parity allocation. It is the natural correlation-aware sibling of H653 and a cleaner null than
the SLSQP-Sharpe H650 — a return-blind allocation that still uses the deployable leverage mechanism.

**Expected behavior:** ERC should sit between inverse-vol (H653) and the optimized H650 — better than naive
inverse-vol because it accounts for the L/S and parity sleeves' low correlations, but below the
return-optimized blend because it ignores expected returns entirely.

**Success:** deployable Sharpe ≥ the inverse-vol baseline H653 and clearing the keep bar, with near-equal
realized risk contributions (a correct ERC solve). **Falsify:** ERC lands below inverse-vol, or fails the
keep bar → correlation-aware risk parity adds nothing over naive inverse-vol here.

**Result:** appended after run.
