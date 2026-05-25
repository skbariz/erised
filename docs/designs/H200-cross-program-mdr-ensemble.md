# H200: CrossProgramMDRPortfolio

## Hypothesis
The H150-H199 program tested 50 cross-sectional equity strategies. The H001-H100 program tested 100 asset-allocation strategies. The two program families are inherently decorrelated: asset-allocation strategies trade SPY/TLT/GLD while cross-sectional strategies trade relative positions within the S&P 500 equity universe.

Combining the best of both families via MDR optimization should produce a portfolio that exceeds both individual program champions:
- H095 (intra-program MDR over H001-H100): Sharpe 1.133, MaxDD -7.16%
- Best H150-H199 individual: H192 (PCAOUMeanReversion): Sharpe 1.175, MaxDD -24.97%

Target: Sharpe 1.4-1.7, MaxDD < 15%

## Design
- H001-H100 component pool: same 37 strategies as H095 + H097 + H098
- H150-H199 component pool: 9 curated price-based cross-sectional strategies
  (PCA stat arb, reversal, defensive, composite — maximally decorrelated from H001-H100)
- Optimization: SLSQP MDR (same as H095); feasible in ~2 hours total runtime

## Selected H150-H199 components
Criteria: price-only (no fundamental data), Sharpe ≥ 0.95, diverse families:
1. H157 VolScaledMomentum (1.079) — vol-scaled cross-sectional
2. H175 BettingAgainstBeta (1.088) — defensive/low-beta equity
3. H185 WeeklyReversalSectorNeutral (1.103) — short-term reversal
4. H186 ResidualWeeklyReversal (1.095) — residual reversal
5. H189 ResidualOUMeanReversion (1.025) — OU stat arb
6. H191 PCAResidualReversal (1.135) — PCA-based stat arb
7. H192 PCAOUMeanReversion (1.175) — PCA OU stat arb (highest Sharpe H150+)
8. H194 LongShortReversal (0.829) — L/S (near market-neutral)
9. H199 MultiSignalComposite (1.020) — blended signal

## Expected regime behavior
The cross-sectional strategies (especially PCA stat arb and reversal) are expected to be
near-market-neutral and decorrelated from the H001-H100 risk-parity strategies. The MDR
optimizer should concentrate weight on H078 (seasonal risk parity, +0.008 H001 corr) and
H192/H191 (PCA stat arb) as universal diversifiers.

## Success criteria
- OOS Sharpe > 1.1 (meaningfully above H095's 1.133 baseline)
- MaxDD < 15% (improved vs H095's -7.16%)
- CI lower bound > 0
- Cross-program diversification demonstrably adds value vs H095 alone

## Falsification
- If H150-H199 components get near-zero MDR weights, cross-sectional equity adds no diversification
- If MaxDD is worse than H095, the equity component draws down in sync with risk-parity strategies
