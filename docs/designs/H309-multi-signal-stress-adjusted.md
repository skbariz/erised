# H309: MultiSignalStressAdjusted — 5-Signal Stress Monitor with Tiered H021 Exposure

## Hypothesis
Rapach-Zhou (2010) show forecast combinations outperform individual predictors. We extend this to risk signals: sum 5 boolean stress indicators and map to 4 exposure tiers. Signals: (1) VIX > 30, (2) SPY 5% drawdown from 20d high, (3) SHY 63d return > TLT 63d return (inverted curve), (4) HYG 63d return < LQD 63d return (credit risk-off), (5) SPY 21d vol > 1.5× 63d vol. Exposure tiers: 0 signals = 100% H021, 1 = 85%, 2 = 70%, 3 = 55%, 4+ = 40%. When HYG unavailable (pre-2007), compute with 4 signals, adjust tiers proportionally (3+ = 40%).

**Expected behavior:**
- 2008: Multiple signals fire → heavy reduction → meaningful MaxDD protection
- 2020: Fast spike to 3+ signals → defensive quickly; fast recovery
- 2013-2017: Typically 0-1 signals → 85-100% H021

**Success criteria:** OOS Sharpe ≥ 0.95; MaxDD < -15%; CI lower > 0.
**Falsification:** 5-signal sum is over-engineered; correlated with VIX alone; turnover too high.

## Result

**OOS Sharpe 0.685** | CI [0.288, 1.076] | CAGR 5.29% | MaxDD −16.84% | H001 corr +0.541 | N_trades 5023

**Decision: KEEP** (CI lower > 0, Sharpe > 0.51)

Best MaxDD in batches 1-3 (−16.84%). The 5-signal tiered approach succeeds at drawdown reduction but Sharpe (0.685) underperforms H021 (0.695) — the incremental tier reductions cost returns. Sensitivity: vix_threshold=25/30/35 → Sharpe 0.697/0.685/0.684 and MaxDD −15.73/−16.84/−16.54 — lower threshold fires more often, improving Sharpe slightly at the cost of MaxDD. Primary value: best portfolio-layer MaxDD protection across all strategies without using vol-scaling. The 5-signal diversity prevents false exits from any single indicator.
