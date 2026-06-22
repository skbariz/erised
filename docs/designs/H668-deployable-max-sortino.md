# H668 — DeployableMaxSortino668

**Signal:** Deployable counterpart of H650, but the pool weights are fit (SLSQP, long-only, sum-to-1,
MDR warm-start) to maximize the **Sortino ratio** — annualized mean ÷ downside deviation
`sqrt(mean(min(r,0)²))·√252` — instead of the Sharpe ratio's total volatility. The same 1-day-lagged
interior-grid leverage as H650 is then applied (threshold {15,16,17} × window {20}, plain + asymmetric).

**Why it should work / what it tests:** Sharpe penalizes upside and downside dispersion symmetrically, so the
H650 weight solve trims sleeves with large *positive* swings just as hard as those with large losses. Sortino
only penalizes the downside, so the allocation is free to keep convex, right-skewed sleeves (trend-following,
the crisis-alpha parity sleeves) that contribute good days without being charged for them. On a book whose
edge is asymmetric — small steady gains plus occasional crisis payoffs — maximizing return-per-unit-downside
should produce a higher Calmar and a lower MaxDD than the symmetric Sharpe fit.

**Expected behavior:** for near-Gaussian sleeves downside-dev ≈ σ/√2, so Sortino-fitted weights should track
H650 closely; divergence should appear as a tilt toward the positively-skewed trend/crisis sleeves, improving
MaxDD/Calmar at a roughly comparable Sharpe.

**Success:** an interior, lag-fitted config with deployable Sharpe ≥ 1.7 and Calmar/MaxDD at least matching
the Sharpe-optimal H650, weight-stable across SLSQP starts. **Falsify:** Sortino-fitting underperforms H650 on
both Sharpe and Calmar, or no interior config clears Sharpe 1.7 → downside-only fitting adds nothing here.

**Result (KEEP).** Config t15w20. Sharpe **1.895** [1.50, 2.28], Calmar 1.221, Sortino **2.006** (batch high),
CAGR 8.47%, MaxDD −6.94%, H001 corr +0.506, folds 1.38–2.99. Fitting return-per-downside lands just below the
Sharpe-optimal H650 (1.979, −0.084) and the MAD fit (H658, 1.953, −0.058) on Sharpe, but delivers the batch's
**best Sortino (2.006)** — exactly as expected when the objective targets downside dispersion directly — with a
tight −6.94% MaxDD and the lowest H001 corr among the return-aware fits (+0.506). The downside-only penalty
keeps the convex trend/crisis sleeves the symmetric Sharpe solve trims, but on this pool that buys a better
**Sortino**, not a better Sharpe. 5-start weight-stable (max spread 0.000). A strong, deployable keeper.
