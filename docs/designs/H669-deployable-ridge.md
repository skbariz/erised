# H669 — DeployableRidge669

**Signal:** Deployable counterpart of H650 with the optimized weights **L2-shrunk toward equal weight**: fit
`w` (SLSQP, long-only, sum-to-1, MDR warm-start) to maximize `Sharpe(pool·w) − λ·‖w − 1/n‖²`. The ridge
penalty pulls the Sharpe-optimal solution back toward the 1/55 equal-weight book by a controllable amount.
Canonical `λ = 0.5`; sensitivity reported over `λ ∈ {0, 0.25, 0.5, 1.0, 2.0}`. The same 1-day-lagged
interior-grid leverage as H650 is then applied (threshold {15,16,17} × window {20}, plain + asymmetric).

**Why it should work / what it tests:** the unconstrained Sharpe solve (H650) is the classic Markowitz
error-maximizer — it loads onto whichever sleeves had the luckiest in-sample mean/covariance estimate, which
is exactly the weight configuration least likely to survive OOS. Shrinking toward 1/n is the portfolio analog
of ridge regression: it trades a little in-sample Sharpe for weights that are closer to the maximally-robust
equal-weight book. This penalizes the *weights* directly, distinct from shrinking the covariance (H654) or
penalizing turnover vs the previous book (H664).

**Expected behavior:** as λ rises, in-sample Sharpe falls monotonically while the weight vector flattens toward
1/n and the realized OOS fold-Sharpe dispersion tightens. The interesting outcome is a λ where deployed Sharpe
is *flat or improved* — i.e. the dropped in-sample Sharpe was overfit.

**Success:** a non-trivial λ whose deployable Sharpe ≥ H650 (or within noise) at clearly flatter weights and
tighter fold dispersion, clearing the keep bar. **Falsify:** every λ > 0 strictly lowers deployable Sharpe →
the H650 solution is not overfit on this pool and shrinkage only costs return.
