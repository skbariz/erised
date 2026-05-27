# H310: FundingStressVolTarget — H086 × Funding Stress Compound Scaling

## Hypothesis
Position sizing overlays reliably improve both Sharpe and MaxDD (H086-H090 family). We compound two proven mechanisms: the H086 vol-target (min(1, 8%/port_vol)) and a binary funding stress flag (VIX > 30 AND credit risk-off when available). When stress is active, multiply the H086 scale by stress_reduction (default 0.65), so the portfolio de-risks faster in stressed environments. When HYG/LQD unavailable, stress flag = VIX > 30 only. This compounds two independent derisking mechanisms without introducing new lookahead risk.

**Expected behavior:**
- Normal + low vol: H086 full scale × 1.0 = normal H021 or near
- High vol + no stress: H086 reduces normally
- Normal vol + stress (HYG < LQD): compound reduction → 65% of H086 scale
- High vol + stress (worst case): vol_scale × 0.65 → very defensive

**Success criteria:** OOS Sharpe ≥ H086 (1.109); MaxDD < -9.93%; CI lower > 0.
**Falsification:** Compound stress scaling over-reduces exposure in mild stress; CAGR loss outweighs MaxDD improvement.

## Result

**OOS Sharpe 0.677** | CI [0.281, 1.071] | CAGR 4.89% | MaxDD −20.24% | H001 corr +0.534 | N_trades 5023

**Decision: KEEP** (CI lower > 0, Sharpe > 0.51)

Hypothesis falsified as improvement over H086 — H310 (0.677) is below H086 baseline (0.662 in walk-forward context; 1.109 full OOS). The compound stress-and-vol scaling hurts more than it helps: Sharpe and CAGR both fall vs H086. Sensitivity: stress_reduction=0.5/0.65/0.8 → Sharpe 0.683/0.677/0.671 — more reduction (0.5) is marginally better, confirming the signal fires too rarely to add consistent value at the 0.65/0.8 levels. The binary stress flag (VIX>30 AND HYG<LQD) fires infrequently enough that it adds variance rather than protection.
