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

**Result (KEEP, partial falsification).** Config t15w20, equal-weight blend of six champion allocations
(Sharpe/Sortino/CAGR-floor/CVaR/ERC/LW-GMV). Sharpe **1.887** [1.49, 2.27], Calmar 1.309, Sortino 1.970,
CAGR 9.93%, MaxDD −7.58%, H001 corr +0.527, folds 1.11–2.65. Component Sharpes ranged 1.41 (ERC) to 1.98
(H650); the blend lands **below** the simple Sharpe H650 on both Sharpe (−0.092) and Calmar (−0.023) — so
the falsification clause fires: averaging in the two **return-blind** baselines (ERC 1.414, GMV 1.516)
dilutes more than objective-diversification adds. The lesson mirrors the program's composite-dilution finding
([[feedback_composite_signal_dilution]]): blend only the return-optimized objectives, not return-blind
allocators. Still a robust, fully-executable book far above the keep bar — just not frontier-leading.
