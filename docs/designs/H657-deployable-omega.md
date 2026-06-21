# H657 — DeployableOmega657

**Signal:** Deployable counterpart of H650, but the pool weights are fit to maximize the **Omega ratio at a
0% daily threshold** — Ω(0) = Σ(gains) / Σ(|losses|), the probability-weighted ratio of upside to downside
across the *whole* return distribution — on the **1-day-lagged** VIX-momentum regime, over the interior-only
grid (threshold {15,16,17} × window {20}, plain + asymmetric). Full battery: 5-start weight stability,
threshold sensitivity, gap vs H650.

**Why it should work:** Sharpe collapses the distribution into its first two moments; Omega uses every
moment by integrating gains and losses about a threshold. On a pool whose sleeves differ in skew/kurtosis
(L/S reversal, calendar, parity), Omega-fitting should reward sleeves with favorable higher moments that
Sharpe ignores — a distinct, fully-distributional point on the deployable frontier.

**Expected behavior:** in calm regimes Omega ≈ a monotone transform of Sharpe (near-Gaussian), so weights
should resemble H650; the difference should appear in crisis folds where skew dominates.

**Success:** an interior, lag-fitted config with deployable Sharpe ≥ 1.7 and Omega ≥ the Omega of the
Sharpe-optimal H650, weight-stable. **Falsify:** Omega-fitting underperforms H650 on both Sharpe and Omega,
or no interior config clears Sharpe 1.7 → higher moments add nothing here.

**Result:** appended after run.
