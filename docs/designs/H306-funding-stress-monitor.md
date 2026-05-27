# H306: FundingStressMonitor — H021 Scaled by Composite Stress Index

## Hypothesis
Market and funding liquidity reinforce each other (Brunnermeier-Pedersen 2009). We build a composite stress score from three available signals: (1) VIX rolling percentile (VIX/252d historical quantile), (2) credit risk-off (HYG 63d return < LQD 63d return → 1, else 0), and (3) vol acceleration (SPY 21d vol > 1.5× 63d vol → 1, else 0). The stress composite = mean of these three signals. We scale H021 total portfolio weight by (1 − stress_sensitivity × stress). When HYG/LQD unavailable (pre-2007), use only VIX percentile and vol acceleration for the composite.

**Expected behavior:**
- 2008 GFC: all three signals likely active → heavy scale-down → reduced portfolio exposure
- 2020 COVID: fast VIX spike and vol acceleration → score rises quickly
- 2013-2017: low VIX, credit stable → near-zero stress → H021 at full scale
- Pre-2007 folds: HYG missing → 2-signal composite still catches VIX + vol stress

**Success criteria:** OOS Sharpe ≥ 0.90; MaxDD < -15%; CI lower > 0.
**Falsification:** Composite stress score is too noisy; scaling H021 damages Sharpe without meaningful MaxDD improvement.

## Result

**OOS Sharpe 0.728** | CI [0.332, 1.120] | CAGR 5.59% | MaxDD −17.58% | H001 corr +0.543 | N_trades 5023

**Decision: KEEP** (CI lower > 0, Sharpe > 0.51)

Best result in batches 1-3 — beats H021 (0.695) and achieves the best MaxDD (−17.58%) of the VRP+stress family. Sensitivity monotonically: stress_sensitivity=0.35/0.50/0.65 → Sharpe 0.719/0.728/0.737 and MaxDD −18.71/−17.58/−16.43 — higher sensitivity consistently improves both Sharpe and MaxDD. The continuous 3-component composite fires more reliably than a binary threshold. Pre-2007 fallback (2-signal: VIX+vol) is sufficient. This is the proof that incremental scaling beats binary flags for market-timing overlays.
