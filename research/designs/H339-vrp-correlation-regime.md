# H339: VRPCorrelationRegime -- CorrelationRiskParityHybrid (H097) + VRP boost in normal regime

## Hypothesis
H097 (CorrelationRiskParityHybrid) uses the sign of the rolling 60d SPY-TLT correlation as a crisis detector: positive correlation (both assets falling together) → crisis override to 90% TLT. This already handles drawdown protection well. This strategy adds a VRP boost in the normal regime (negative correlation): when VRP is above its rolling 252d median, increase SPY weight by 10% (capped at 1.0). The VRP signal in normal (non-crisis) conditions indicates whether the market is adequately compensating for variance risk. A VRP above median in a non-crisis environment is a positive signal for holding more equity exposure. The crisis detection (correlation) and the return-enhancing signal (VRP) are designed to operate in different market phases without interfering.

**Expected behavior:**
- 2000-2002: Crisis correlation triggers → VRP boost never fires in crisis; protection from H097 intact
- 2008-2009: Correlation turns positive; no boost; protective override active
- 2013-2017 low vol: Normal regime throughout; VRP consistently above median → 10% boost applied most of the time → modest return improvement
- 2020 COVID: Correlation turns positive in Feb-March → crisis mode; VRP boost irrelevant during crisis; post-crisis normal regime restored

**Success criteria:** OOS Sharpe ≥ H097 (0.992); MaxDD ≤ -0.10%; CI lower > 0; fold consistency.
**Falsification:** 10% boost is too small; or VRP above median in normal regime does not predict outperformance (noise in non-crisis conditions).

## Results
*[To be filled after backtest]*
Decision:
