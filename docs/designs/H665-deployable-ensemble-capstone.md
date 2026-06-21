# H665 — DeployableEnsembleCapstone665

**Signal:** The **Set14 capstone**. At the frontier-champion config (t15w20, 1-day-lagged regime), fit the
six deployable champion allocations — Sharpe (H650), Sortino (H651), CAGR-with-floor (H655), return-to-CVaR
(H656), equal-risk-contribution (H659), and Ledoit-Wolf GMV (H660) — and **equal-weight blend their weight
vectors** into one deployable allocation, then apply the shared lagged-interior leverage once. A single,
fully-executable fund-of-objectives.

**Why it should work:** each objective is right about a different risk — variance, downside, growth-floor,
tail, risk-contribution balance, covariance noise — and each is wrong in its own way out of sample. Averaging
the *allocations* (not the streams) diversifies across objective-specification error, the same way H610
blended champion streams. The blend should land near the top of the deployable Sharpe cluster while
inheriting the best of each component's drawdown/tail behavior, with no single objective's idiosyncratic
overfit dominating.

**Expected behavior:** the blended weights should be less concentrated than the pure Sharpe fit (the ERC/GMV
components pull toward diversification), giving Sharpe close to H650 with a Calmar/Sortino at least matching
the better single objectives.

**Success:** deployable Sharpe within ~0.05 of the best single Set13/14 objective with a Calmar ≥ the median
component — i.e. the blend is on or above the deployable frontier, not a dilutive average. **Falsify:** the
blend underperforms the simple Sharpe H650 on *both* Sharpe and Calmar → averaging objectives only dilutes.

**Result:** appended after run.
