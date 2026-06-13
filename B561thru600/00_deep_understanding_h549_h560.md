# Deep Understanding: H549–H560 (10 Most Recently Kept)

## The arc in one sentence
MDR maximized diversification ratio but was blind to leverage. Multi-objective SLSQP (H551) found a
better composite portfolio. Applying vol-targeting leverage to that portfolio (H556) crossed 1.79.
Optimizing the *leveraged* objective directly (H558/H559) unlocked the full gain: 1.879 Sharpe and
13.08% CAGR — both program records. Concentrated-pool testing (H560) confirmed the full pool is
optimal for this objective class.

---

## Strategy-by-strategy breakdown

### H549 — Set7GrandCrossMDR (Sharpe 1.546)
**Mechanism:** SLSQP MDR over 52 strategies — clean "all-12-fold" pool only (strategies with no
zero-Sharpe folds). AA backbone + Set6 stars (H412/H415/H429/H447) + L/S triple (H471/H472/H473)
+ Set7 CS stars (H499/H505/H508/H530/H535/H537/H538/H540).

**Key finding:** H473 (LongShortPCAReversal) at 23.6% dominates — its market-neutral daily
reversal signal has near-zero H001 corr and provides the diversification ratio most of its weight.
H537 earns 7.8% — that's the best CS strategy not in the L/S family.

**Weight pattern:** Traditional MDR signature — H043 10%, H068 9%, H016 10%, H076 7.5%, H065 6.4%.
H473 towers over all. This is the "MDR likes volatility premium and calendar signals" pattern.

**Weakness:** Fold 6 (2014-2016) at 0.785 and fold 10 (2020-2022) at 0.897 are near-zero — EDGAR
timing noise hits fold 6; late-2021 high equity vol collapses CAGR in fold 10.

---

### H550 — UltimateTerminalMDR550 (Sharpe 1.593)
**Mechanism:** Expanded MDR — same 52 strategies as H549 plus H265/H268/H189/H157/H194 for 67
total. Intended as the Set7 terminal champion.

**Key finding:** Expanding from 52→67 strategies gives +0.047 Sharpe. Each marginal addition is
small but they compound. H265 (SustainabilityFilteredProfitability) earns 6.2% — enters for the
first time at meaningful weight.

**Weight pattern:** Nearly identical to H549 (H473 22.8%). The new strategies share weight evenly
at 1-6%. No single new strategy dominates. MDR found H265 has useful diversification properties
vs the calendar/vol-premium anchor strategies.

**Weakness:** Same fold 6/10 issues as H549.

---

### H551 — MultiObjectiveMDR (Sharpe 1.741) ★ ARCHITECTURE BREAKTHROUGH
**Mechanism:** SLSQP maximizing composite = 0.33×Sharpe + 0.33×Calmar + 0.33×Sortino on the
same 55-strategy pool. Warm-started from MDR solution. Discovers a completely different portfolio.

**The discovery that changed everything:** The composite optimizer finds H065/H076 as dominant
(27%/26%), not H043/H468/H016 as MDR does. WHY: MDR maximizes diversification ratio —
essentially it concentrates on *uncorrelated* strategies. H043 (sell-in-may) has very low
correlation with everything and earns MDR weight. But H043 contributes little to Calmar because
it sits in TLT for 6 months — low CAGR during those periods.

H065 (EqualWeightDefensiveFactor) and H076 (FourAssetRiskParity) earn Calmar-friendly weight:
they produce consistent positive returns in most regimes without the equity-crash exposure.

**H473's role shifts:** In MDR, H473 earns 22-24% because it's uncorrelated. In multi-objective,
H473 earns 16.5% — still important but the composite objective no longer *concentrates* on
uncorrelation alone; it also needs the strategy to contribute CAGR.

**The pure-Sharpe surprise:** Grid results show pure-Sharpe base (w1=1.0) gives 1.895 base Sharpe
at -8.37% MaxDD. This is higher than the multi-objective solution (1.741) but with deeper
drawdown. The optimizer trades 0.154 Sharpe for -3.96pp MaxDD reduction. This is the tradeoff
that Set8 Batch 2-3 will fully explore.

**Weight pattern:** H065 26.9%, H076 25.8%, H473 16.5%, H157 15.0%, H016 7.4%.
Only 5 strategies hold >5% weight — very concentrated.

---

### H552 — MaxDDConstrainedMDR (Sharpe 1.584)
**Mechanism:** MDR with MaxDD penalty (penalty=1.0) over expanded 58-strategy pool (H543's 55 +
H537/H538/H540). Intended to find a lower-MaxDD MDR solution.

**Key finding:** Penalty=1.0 barely changes the MDR solution because MDR already achieves
MaxDD ≤ 10% naturally (it maximizes diversification which also limits drawdown). The solution is
effectively identical to MDR on the expanded pool.

**Weight pattern:** Near-identical to H549/H550 MDR pattern. H473 23.2%.

**Why it underperforms H551 (1.584 vs 1.741):** MDR's diversification-ratio objective doesn't
align with Sharpe/Calmar. H551's composite objective does. The 58-strategy expansion helps
slightly but can't bridge the objective-function gap.

---

### H553 — RatchetOnH552 (Sharpe 1.692, MaxDD -4.22%)
**Mechanism:** Ratchet overlay (dd_limit=0.25) on H552's MDR portfolio. Leverage = 1× normally,
cuts to (1-dd/dd_limit)× when portfolio drawdown exceeds threshold.

**Key finding:** MaxDD drops from -6.77% → -4.22% — the ratchet fires when needed. But the
ratchet rarely fires on an MDR portfolio that already has MaxDD -6.77%, so the average leverage
stays near 1×. This explains why Sharpe improves only modestly (1.584 → 1.692) while MaxDD
improves dramatically.

**Important insight for leverage research:** Ratchet is a risk-reducing overlay on an already
conservative portfolio. It contributes more to MaxDD reduction than to Sharpe improvement. The
correct use case for ratchet is on a *higher-risk* portfolio where it fires more frequently.

---

### H556 — DynamicLeverageVolTarget (Sharpe 1.790, MaxDD -5.79%) ★ PRIOR CHAMPION
**Mechanism:** leverage_t = min(2.0, 6.5%/realized_vol_{t-1,20d}) applied to H551's fixed weights.
Avg realized leverage 1.23× — vol-targeting de-levers frequently because H551's portfolio has
moderate realized vol.

**Key finding:** +0.048 Sharpe over H551 base (1.790 vs 1.741) with only +1.38pp MaxDD worsening.
The vol-targeting mechanism is "Sharpe-preserving" — it scales up gently (1.23× avg) and scales
down proactively when vol rises, avoiding the worst drawdown days.

**Crucially:** H551's weights were NOT optimized for leveraged deployment. They were optimized
for the base portfolio. The vol-target applies leverage to a portfolio designed for 1×. This is
the gap that H558/H559 will close.

**Weight pattern:** Fixed H551 weights — H065 26.9%, H076 25.8%, H473 16.5%, H157 15.0%.

---

### H557 — RatchetLeverage (Sharpe 1.701, MaxDD -6.48%, CAGR 10.90%)
**Mechanism:** leverage_t = 1 + 0.5×max(0, 1 - dd/15%) on H551's fixed weights. Max leverage 1.5×.
Avg realized leverage 1.47× — the ratchet almost never fires because H551's base MaxDD is only
-4.40%. Strategy runs at near-max leverage nearly all the time.

**Key finding:** CAGR jumps to 10.90% (+3.36pp vs H551 base) because avg 1.47× leverage.
But Sharpe falls: 1.701 vs H556's 1.790 (-0.089). Running at near-constant 1.47× adds systematic
risk without the proactive de-risking that vol-targeting provides.

**Design lesson:** Ratchet leverage is "CAGR-first, Sharpe-second." Use it when the objective
function is CAGR/Calmar and drawdown control is secondary. For Sharpe optimization, vol-target
is strictly better because it fires proactively (daily) rather than reactively (only after a
drawdown has started).

---

### H558 — CalmarOptimizedMDR (Sharpe 1.697, CAGR 11.89%, Calmar_lev 1.802) ★ CALMAR CHAMPION
**Mechanism:** SLSQP with leveraged-Calmar as objective — optimize Calmar of the vol-targeted
portfolio directly (not the base portfolio). Same 55-strategy pool, MDR warm-start.

**THE KEY FINDING: "optimize the deployed metric":**
| Objective | Cal_base | Sh_lev | Cal_lev |
|---|---|---|---|
| Pure Calmar | 1.863 | 1.640 | 1.429 |
| Leveraged Calmar | 1.280 | 1.697 | 1.802 |

The portfolio that maximizes BASE Calmar collapses under leverage (Cal_lev 1.429). The portfolio
that optimizes LEVERAGED Calmar has lower base Calmar (1.280) but far better Cal_lev (1.802).
Why: the leveraged-Calmar optimizer co-adapts weights to the vol-targeting mechanism. It selects
strategies whose volatility dynamics PAIR WELL with daily vol-targeting leverage.

**Weight shift vs H551:** H065 INCREASES to 30% (from 26.9%), H076 DECREASES to 15.6% (from 25.8%),
H473 INCREASES to 28.6% (from 16.5%). The optimizer concentrates heavily in H473 because:
- H473's market-neutral return stream has vol that's nearly uncorrelated with SPY vol
- When SPY vol spikes (→ de-lever), H473's contribution is unaffected
- This makes H473 a "vol-targeting-stable" component

**Fold pattern:** All 12 positive, range [1.071, 2.328]. Fold 11 (2022-2024) is the floor at 1.071.

---

### H559 — LeveragedSharpeMDR (Sharpe 1.879, CAGR 13.08%) ★ PROGRAM CHAMPION
**Mechanism:** SLSQP with leveraged-Sharpe as objective — optimize Sharpe of vol-targeted portfolio.
Same 55-strategy pool, MDR warm-start.

**The breakthrough:**
| Objective | Sh_lev | Cal_lev | CAGR_lev | DD_lev |
|---|---|---|---|---|
| H556 ref (H551 weights + post-hoc lev) | 1.683 | 1.520 | 11.70% | -7.70% |
| Lev-Sharpe (H559) | **1.879** | 1.344 | **13.08%** | -9.73% |
| Lev-Sortino | 1.874 | 1.397 | 13.20% | -9.45% |
| Lev-Composite (Sh+Cal+So) | 1.762 | **1.735** | 12.58% | **-7.25%** |

**Weight pattern — NEW vs H558:**
H076 returns to 25.4% (H558 had 15.6%), H065 drops to 21.1% (H558 had 30%), H473 drops to 15.0%
(H558 had 28.6%), H265 ENTERS at 14.7% (H558 had 1.4%), H191/H192 PCA reversion ENTER at 5.6%/7.3%.

**Why H265 matters:** SustainabilityFilteredProfitability produces quarterly-turnover returns with
very low autocorrelation. Under vol-targeting leverage, autocorrelation = variance stability risk.
High-autocorrelation strategies (trend followers) can have vol that persists across the vol-
targeting lookback window. H265's low-autocorrelation pairs exceptionally well with daily vol-
targeting — it reduces the "persistence risk" in the portfolio's vol profile.

**Why H191/H192 enter:** PCA-based residual mean-reversion (daily signal, daily resolution).
These strategies produce returns that are anti-correlated with short-term SPY moves. Under
vol-targeting, when SPY vol spikes (→ reduce leverage), H191/H192 tend to be POSITIVE (reverting
from the spike). This creates a natural hedge against the cost of de-leveraging.

**H001 corr drops to +0.451** (H556 was +0.612, H558 was +0.519). The lev-Sharpe objective
achieves lower H001 correlation than any prior leveraged ensemble — the optimizer found that
reducing directional equity exposure HELPS Sharpe under vol-targeting.

**The tradeoff at the top:** H559 accepts deeper MaxDD (-9.73%) than H556 (-5.79%) to gain
+0.089 Sharpe. The MaxDD-Sharpe tradeoff remains open — Lev-Composite offers 1.762 Sharpe with
only -7.25% DD. Neither the Sharpe frontier nor the MaxDD-constrained frontier has been fully
explored.

---

### H560 — ConcentratedPoolMDR (Sharpe 1.697, Calmar_lev 1.802)
**Mechanism:** Lev-Calmar objective applied to top-20, top-30, and full-55 pools. Tests H516's
finding that smaller focused pools beat full pools for MDR.

**Finding:** Full-55 WINS:
| Pool | Sh_lev | Cal_lev |
|---|---|---|
| Full-55 | 1.697 | **1.802** |
| Top-20 | 1.673 | 1.764 |
| Top-30 | 1.653 | 1.727 |

H516's finding (20-strategy focused pool cracked MDR ceiling) does NOT generalize to leveraged
SLSQP. Why: H516's MDR worked in unleveraged context where diversification was paramount — a
focused pool of uncorrelated strategies maximizes diversification ratio. In leveraged SLSQP,
the optimizer needs strategies with diverse VOL PROFILES, not just diverse return profiles.
The full pool provides more vol-profile diversity (market-neutral CS with different turnover
frequencies) that concentrated high-Sharpe pools lack.

**Also confirms H558:** H560's full-pool lev-Calmar solution is identical to H558 (same weights,
same result). The result is reproducible and robust.

---

## Cross-cutting patterns

### Weight concentration across the 10 strategies

| Strategy | H473 | H065 | H076 | H016 | H043 | H265 | H191/H192 |
|---|---|---|---|---|---|---|---|
| H549 MDR | 23.6% | 6.4% | 7.5% | 10.4% | 10.0% | — | — |
| H550 MDR+exp | 22.8% | 6.4% | 7.6% | 10.2% | 9.6% | 6.2% | — |
| H551 Multi-obj | 16.5% | **26.9%** | **25.8%** | 7.4% | 1.9% | 0.5% | 1.3% (H192) |
| H552 MDR+pen | 23.2% | 6.4% | 7.5% | 10.1% | 9.6% | 6.3% | — |
| H556 (H551 base) | 16.5% | 26.9% | 25.8% | 7.4% | 1.9% | 0.5% | 1.3% |
| H558 Lev-Cal | **28.6%** | **30.0%** | 15.6% | 3.3% | — | 0.9% | — |
| H559 Lev-Sh | 15.0% | 21.1% | **25.4%** | 5.6% | — | **14.7%** | **12.9%** |
| H560 Lev-Cal | 28.6% | 30.0% | 15.6% | 3.3% | — | 0.9% | — |

**H473 is a chameleon:** 15-29% depending on objective. MDR maximizes diversification so H473's
near-zero H001 corr earns massive weight. Lev-Calmar also rewards H473 (market-neutral, low vol
persistence). Lev-Sharpe finds 15% is enough — H076/H265/H191/H192 displace it.

**H265 is the "lev-Sharpe specialist":** Near-zero weight in MDR/multi-obj, 14.7% in lev-Sharpe.
The quarterly-rebalance, low-autocorrelation fundamental signal is uniquely valuable under
vol-targeting. It's the most undertested strategy in the pool.

**H191/H192 breakthrough:** PCA mean-reversion strategies enter only under lev-Sharpe. Daily
resolution, anti-SPY-momentum during vol spikes. This is a new class of "vol-targeting-compatible"
signal that should be explored more.

### The evolving understanding of leverage

1. **H553:** Ratchet as passive risk reducer. Rarely fires on low-DD base. Contributes to MaxDD
   control but not Sharpe.
2. **H556:** Vol-target as proactive Sharpe preserver. Fires daily. +0.048 Sharpe over base.
3. **H557:** Ratchet as CAGR amplifier. Fires rarely → near-constant leverage → CAGR up,
   Sharpe down.
4. **H558/H560:** Lev-Calmar as co-optimizer. Weights co-adapt to leverage mechanism.
5. **H559:** Lev-Sharpe as co-optimizer. +0.089 Sharpe over post-hoc application.

**The fundamental insight:** Leverage is not a post-processing step. It's an integral part of
the portfolio construction. When you optimize for the deployed metric (Sh_lev, Cal_lev), the
optimizer finds strategies that pair well with the leverage mechanism — not just strategies that
have high standalone Sharpe.

### Fold fragility

| Fold | Period | Weakest strategy | Why |
|---|---|---|---|
| 6 | 2014-2016 | H552 (1.010) | EDGAR gap collapses cross-sectional |
| 10 | 2020-2022 | H549/H550 (~0.9) | Post-COVID high equity vol, MDR-heavy |
| 11 | 2022-2024 | H558/H560 (1.071) | Rate shock: TLT/GLD both negative |
| 5 | 2010-2012 | H559 (1.157) | European debt crisis, vol-target de-levers |

H559's minimum fold (1.157) is higher than H549/H550's minimum (0.785/0.897) — the lev-Sharpe
optimization explicitly improves the worst folds as part of maximizing Sharpe.

---

## Open questions that drive H561-H600

1. **Is H559 (1.879) the true global optimum?** MDR warm-start may have found a local optimum.
   Multi-restart or global search (differential evolution) could reveal a better solution.

2. **What is the exact MaxDD-Sharpe tradeoff frontier?** Lev-Composite gives 1.762 Sharpe at
   -7.25% DD. Lev-Sharpe gives 1.879 at -9.73%. What's the Pareto-optimal curve between them?

3. **Can higher leverage (3.0×) reach Sharpe 2.0?** H559's avg realized 1.23× is far below
   the 2.0× cap. If we push cap to 3.0× with target_vol 5%, avg leverage might reach 1.5×.

4. **What new individual strategies are vol-targeting-compatible?** H265 and H191/H192 were
   "discovered" by lev-Sharpe. Are there other strategies with quarterly/monthly turnover and
   low autocorrelation that would fill the same role better?

5. **Does the lev-Sharpe objective generalize to new pool members?** H559 used 55 fixed
   strategies. Adding new strategies (especially H191-class daily reversion) could improve.

6. **What's the hierarchical sleeve architecture worth?** Optimizing AA and CS sleeves
   separately, then combining, might find a better tradeoff than joint optimization.

7. **Is the vol-target lookback (20d) optimal?** Faster (5d) reacts to spikes quicker; slower
   (60d) provides more stable leverage. The right lookback depends on the portfolio composition.
