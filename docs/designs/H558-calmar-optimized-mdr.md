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

*(To be filled after run)*
