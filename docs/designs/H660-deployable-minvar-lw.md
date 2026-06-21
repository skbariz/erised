# H660 — DeployableMinVarLW660

**Signal:** A deployable **global-minimum-variance** allocation computed on a **Ledoit-Wolf shrunk
covariance** (analytical shrinkage toward a scaled identity — the intensity is estimated from the data, not
tuned), under the same 1-day-lagged interior-grid leverage as H650. GMV minimizes total portfolio variance
subject to long-only, sum-to-1; LW shrinkage de-noises the 55×55 covariance so the inverse is well-
conditioned. Sweep the interior grid; finalize on the best leverage config; report the LW intensity δ.

**Why it should work / what it tests:** GMV is acutely sensitive to estimation error in the covariance —
sample-covariance GMV chases spurious low-variance combinations and inverts noise. Ledoit-Wolf shrinkage is
the standard fix and has *no* free parameter (δ is the optimal bias-variance trade-off in closed form), so
this is a principled, non-overfit robust-covariance baseline distinct from MDR (H650 warm start), inverse-
vol (H653), and ERC (H659).

**Expected behavior:** the LW intensity δ should be modest-to-large on a 55-asset, ~6k-day panel (estimation
error is real); the GMV weights should concentrate on the low-vol parity/calendar sleeves and deliver the
lowest realized vol of the deployable allocations, with Sharpe near the ERC/inverse-vol baselines.

**Success:** a valid shrinkage δ∈(0,1), trace-preserving shrunk covariance, deployable Sharpe clearing the
keep bar and competitive with ERC (H659). **Falsify:** GMV underperforms the simpler inverse-vol/ERC
baselines, or δ pins to 0/1 → shrinkage adds nothing on this pool.

**Result:** appended after run.
