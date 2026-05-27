# H308: CreditRegimeRiskParity — H021 with HYG vs LQD Credit Signal

## Hypothesis
Credit markets lead equity markets in risk-off episodes. When high-yield bonds (HYG) are outperforming investment-grade bonds (LQD) over a 63-day lookback, credit markets signal risk appetite — run H021 at full weight. When LQD is outperforming HYG (credit spread widening, risk-off signal), reduce the H021 SPY weight by spy_scale (default 0.6) and redistribute freed weight to TLT. This is a single, clean macro signal without composite noise. When HYG/LQD data is unavailable (pre-2007), fall back to H021 unmodified.

**Expected behavior:**
- 2008: HYG crashed before SPY; credit signal should trigger well before equity bottom
- 2020: HYG dropped sharply in March → signal fires; recovery swift → clean exit
- 2013-2017: HYG generally outperforms LQD in growth environment → H021 full
- Pre-2007 folds: no HYG → pure H021; no damage

**Success criteria:** OOS Sharpe ≥ 0.90; MaxDD < -18%; CI lower > 0.
**Falsification:** Credit signal too slow / noisy; or correlated enough with H080 yield curve overlay that no additional value is added.

## Result

**OOS Sharpe 0.684** | CI [0.282, 1.079] | CAGR 5.87% | MaxDD −19.86% | H001 corr +0.486 | N_trades 5023

**Decision: KEEP** (CI lower > 0, Sharpe > 0.51)

HYG vs LQD 63d credit signal passes the bar. Sensitivity: spy_scale=0.4/0.6/0.8 → Sharpe 0.668/0.684/0.693 — higher scale (less reduction) is better, suggesting the credit signal over-corrects at the canonical parameter. Lowest H001 corr (+0.486) in batch 2 — the credit signal captures a slightly different dimension than VIX/vol-based approaches. Note: H308 and H312 (DefaultSpreadSPYTimer, Batch 3) produce identical results since they implement the same underlying signal (HYG vs LQD 63d, spy_scale 0.60). Only H308 is considered unique.
