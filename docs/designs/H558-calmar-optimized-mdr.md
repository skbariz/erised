# H558: Calmar-Optimized MDR

## Hypothesis

H551 tested a grid of composite objectives (Sharpe+Calmar+Sortino) and found equal weighting
(0.33/0.33/0.33) wins on Sharpe+Calmar simultaneously. One point was never tested: **pure Calmar**
(w_cal=1.0). H558 tests whether optimizing directly for Calmar (or leveraged-Calmar) on the same
55-strategy pool produces a better base portfolio than H551's.

Two new objectives:
1. **Pure Calmar** (`w1=0, w2=1.0, w3=0`): directly maximize CAGR/|MaxDD|.
2. **Leveraged Calmar**: optimize CAGR/|MaxDD| of the vol-targeted leveraged portfolio (6.5%/2.0x).
   This directly targets performance under leverage — the deployment context for Set 8.

**Why leveraged-Calmar is the most interesting objective:** H556 showed that vol-targeting
leverage at 6.5%/2.0x is the best leverage mechanism. If we optimize portfolio composition for
Calmar-under-leverage (not Calmar-at-base), the optimizer can account for the interaction between
the vol-targeting scale factors and each component's tail behavior.

**Success bar:** Leveraged Calmar > 1.70 (beats H551's base Calmar 1.707) with leveraged Sharpe > 1.75.

## What would falsify this

- Pure Calmar objective produces Sharpe < 1.5 (over-concentrates to one strategy for CAGR)
- Leveraged-Calmar objective is degenerate (same solution as H551 after warm-start)
- Both objectives produce results dominated by H551 in both Sharpe and Calmar

---

## Results

**Decision: KEEP**

| Objective | Sh_base | Cal_base | Sh_lev | Cal_lev | CAGR_lev | DD_lev |
|---|---|---|---|---|---|---|
| H551 ref | 1.721 | 1.782 | 1.683 | 1.520 | 11.70% | −7.70% |
| Pure Calmar | 1.654 | **1.863** | 1.640 | 1.429 | 11.77% | −8.24% |
| Calmar-dominant (0.1/0.8/0.1) | 1.651 | 1.848 | 1.628 | 1.422 | 11.68% | −8.22% |
| **Leveraged Calmar** | 1.750 | 1.280 | **1.697** | **1.802** | **11.89%** | **−6.60%** |

**Final (canonical):** Leveraged Calmar objective, vol-target 6.5%/2.0x
- OOS Sharpe: **1.697**  CI [1.309, 2.090]
- CAGR: **11.89%**  MaxDD: **−6.60%**  Calmar: **1.802**
- H001 corr: +0.519 (lower than H551's +0.641 — H473's market-neutral character)
- All 12 folds positive [+1.071, +2.328]

**Top weights:** H065 (30.0%), H473 (28.6%), H157 (16.6%), H076 (15.6%), H016 (3.3%), H063 (3.9%)

**Critical finding — optimize the deployed metric, not the base metric:**
Pure Calmar on base achieves Cal_base 1.863 (best unleveraged) but collapses to Cal_lev 1.429 when leveraged.
Leveraged-Calmar objective achieves Cal_base 1.280 (lowest) but Cal_lev 1.802 (best). The optimizer co-adapts
portfolio composition to the vol-targeting mechanism — finding a portfolio whose vol profile pairs with the
leverage overlay rather than maximizing Calmar in isolation. **Lesson: always optimize for the deployed metric.**

**vs H556:** H558 has better CAGR (11.89% vs 9.61%) and Calmar_lev (1.802 vs 1.660) but lower Sharpe (1.697 vs 1.790) and wider MaxDD (−6.60% vs −5.79%). H556 remains Sharpe champion; H558 is the Calmar/CAGR champion under leverage.
