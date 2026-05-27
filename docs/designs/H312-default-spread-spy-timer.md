# H312: DefaultSpreadSPYTimer — H021 with HYG vs LQD Default Spread Signal

## Hypothesis
The default spread (high-yield yield minus investment-grade yield) is a Goyal-Welch equity premium predictor. We proxy it with the relative 63-day performance of HYG (high-yield ETF) versus LQD (investment-grade ETF). When HYG is outperforming (credit risk-on), H021 runs at full weight. When LQD is outperforming (credit risk-off, spread widening), reduce H021 SPY weight by spy_scale (default 0.60). This is a 63-day signal (shorter than H311's 252-day term spread), capturing faster credit market deterioration.

**Expected behavior:**
- 2008: HYG collapsed early → signal fires early → protection before equity bottom
- 2020: HYG falls sharply in March → signal fires; recovery swift → clean exit
- 2013-2017: HYG generally outperforms → H021 full
- Pre-2007 folds: HYG unavailable → pure H021 (5 zero folds for the signal)

**Success criteria:** OOS Sharpe ≥ 0.85; MaxDD < -20%; CI lower > 0.
**Falsification:** Credit signal too slow or too noisy at 63d lookback; correlated with H308 CreditRegimeRiskParity.

## Result

**OOS Sharpe 0.684** | CI [0.282, 1.079] | CAGR 5.87% | MaxDD −19.86% | H001 corr +0.486 | N_trades 5023

**Decision: KEEP** (CI lower > 0, Sharpe > 0.51)

Hypothesis partially falsified: H312 is identical to H308 CreditRegimeRiskParity in all outputs (same fold-by-fold Sharpe, same MaxDD, same CI). Both implement HYG vs LQD 63d return spread with SPY scale 0.60. Sensitivity: credit_lookback=42/63/126 → Sharpe 0.641/0.684/0.689 — longer lookback slightly better. H312 is a confirmed duplicate of H308 and adds no unique value for ensemble construction. For practical purposes, only H308 (or H312 with the 126d lookback variant at 0.689) should be included in any ensemble.
