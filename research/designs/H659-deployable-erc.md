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

**Result (KEEP).** Config t15w20. Sharpe **1.414** [1.02, 1.80], Calmar 0.930, Sortino 1.481, CAGR 7.78%,
MaxDD −8.36%, H001 corr +0.693, folds 0.70–2.22. Realized risk-contribution shares are **0.01818 = 1/55 for
every sleeve** (a textbook-perfect ERC solve). Beats the naive inverse-vol baseline H653 (1.303) by
**+0.111** — correlation-aware risk parity is a genuine improvement over inverse-vol, as hypothesized — but
sits well below the return-optimized H650 (1.979): a return-blind allocation leaves real Sharpe on the table.
ERC is the honest null the optimized objectives must beat, and they do.
