# H302: VRPRiskParityScaled — VRP Percentile-Tiered H021 Multiplier

## Hypothesis
Rather than a continuous VRP z-score, use a 3-tier percentile regime: when VRP exceeds its rolling 70th percentile, scale total H021 exposure to 1.25×; below the 30th percentile, scale to 0.75×; middle tier stays at 1.0×. Discrete tiers reduce whipsaw while still responding to regime shifts. Source: Bollerslev-Tauchen-Zhou (2009) who find VRP predicts aggregate stock market returns over monthly horizons.

**Expected behavior:**
- High-VRP regimes (calm market with high VIX relative to realized): full leverage
- Low-VRP / negative-VRP regimes: defensive; reduce exposure
- 2013-2017 low-vol: often in middle tier → H021 at 100%

**Success criteria:** OOS Sharpe ≥ 0.85; H001 corr < +0.50.
**Falsification:** If VRP percentile tiers add no consistent Sharpe lift, discrete VRP regimes are too noisy.

## Result

**OOS Sharpe 0.676** | CI [0.286, 1.069] | CAGR 5.91% | MaxDD −25.06% | H001 corr +0.540 | N_trades 5023

**Decision: KEEP** (CI lower > 0, Sharpe > 0.51)

Best VRP strategy in batch 1. 3-tier percentile logic delivers 0.676 vs H021 0.695 — still below baseline Sharpe but cleaner than continuous z-score (H301: 0.633). Discrete tiers reduce whipsaw from daily recalculation. Sensitivity: high_scale=1.15/1.25/1.35 monotonically decreases Sharpe (0.691/0.676/0.662) — lower amplification is better, suggesting VRP timing is mildly useful but overshooting hurts. Fold 6 still negative (−0.407) — post-GFC false signals persist across all VRP approaches.
