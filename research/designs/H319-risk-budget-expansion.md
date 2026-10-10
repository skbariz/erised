# H319: RiskBudgetExpansion — Dynamic vol-target expansion/contraction based on VRP and SPY momentum

## Hypothesis
H086 (LowVolTargetPortfolio) showed that applying a fixed 8% vol target to H021 weights produces Sharpe 1.109 and MaxDD -9.93%. The question is whether the vol target itself should vary with macro conditions. This strategy uses two signals to expand or contract the target: (1) VRP > 0 (options market pricing in excess risk premium — favorable for carry strategies), and (2) SPY 252d return > 0 (bullish trend). When both are favorable, expand to 10% target (more risk budget); when either VRP < 0 or VIX exceeds its 60th rolling percentile, contract to 6% target (defensive). The mechanism is identical to H086's vol-scaling but treats the target as state-dependent rather than fixed. This operationalizes the "risk appetite" concept from Bollerslev-Tauchen-Zhou (2009) who show VRP predicts returns.

**Expected behavior:**
- 2000-2002 tech crash: VRP turns negative, SPY 252d negative; 6% target contracts exposure; should outperform H086 during the crash at the cost of some upside in ambiguous periods
- 2008-2009 GFC: VIX at 60th percentile triggering unfavorable; 6% target throughout; conservative positioning
- 2020 COVID: brief unfavorable, rapid shift back to favorable as VRP normalizes; adaptive target catches recovery faster than fixed
- 2013-2017 low vol: VRP positive, SPY up — persistent 10% target expands exposure above H086 baseline; expect higher CAGR

**Success criteria:** OOS Sharpe >= 1.05 (must beat H086 at 1.109 or be close); MaxDD < -12%; CI lower > 0.
**Falsification:** Adaptive target adds no value over H086's fixed 8%; Sharpe ~= 1.109 across all three target levels. Or: 6%/8%/10% all produce the same Sharpe (regime signal adds noise, not information).
