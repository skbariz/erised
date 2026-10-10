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

**Result (KEEP — hypothesis falsified).** Config t15w20, λ=0.5. Sharpe **1.909** [1.51, 2.29], Calmar 1.336,
Sortino 1.995, CAGR **9.01%** (batch high), MaxDD **−6.74%** (batch low), H001 corr +0.520, folds 1.41–2.79.
**The λ-path is the finding and it fires the falsify clause:** deployable Sharpe is *monotonically decreasing*
in λ — 1.926 (λ=0) → 1.921 (0.25) → 1.909 (0.5) → 1.890 (1.0) → 1.862 (2.0) — while ‖w−1/n‖ shrinks
0.436 → 0.251. Shrinking the optimized weights toward equal weight only *costs* return here, so the
H650-style Sharpe solve is **not overfit** on this pool: the in-sample edge it loads onto survives OOS. The
λ=0 point (1.926) confirms the unregularized solve sits in H650's neighborhood. Kept because the canonical
λ=0.5 sacrifices just −0.017 Sharpe for a flatter, more robust book and posts the best CAGR/MaxDD pair in the
batch — a defensible deployment even though, on this pool, regularization is not a free lunch.
