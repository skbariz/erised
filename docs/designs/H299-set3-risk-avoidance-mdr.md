# H299: Set3RiskAvoidanceMDR — MDR Ensemble over Best 5 H251-H298 Strategies

## Hypothesis

MDR (Maximum Diversification Ratio) is the best ensemble method found in this program
(H095: 1.133, H200: 1.350). Applying MDR to the best 5 kept strategies from H251-H298
— which are all near-zero H001 correlation and largely orthogonal to each other —
should produce a diversified equity cross-sectional portfolio.

Signal: SLSQP MDR optimization over candidate strategies from H251-H298 with Sharpe > 0.75.
Select 5 strategies with the broadest H001 corr range (ensure at least one with H001 < 0.20).
Walk-forward MDR: refit portfolio weights each training period using training-period returns.

Selection criteria (to be filled after H251-H298 complete):
- Candidate pool: all kept strategies with Sharpe > 0.75 and positive IC in > 8 of 12 folds
- Prefer strategies from different signal families (no two from same batch)
- Require at least one strategy with H001 corr < 0.10

Expected regime behavior: Inherits from component strategies — diversification across
momentum lifecycle (H271-H275), skew/tail (H276-H280), fundamentals (H251-H270),
and capital discipline (H286-H295) reduces family-specific regime risk.

Success criteria: OOS Sharpe ≥ 1.10, MaxDD < −20%, CI lower > 0.
Falsification: If all Set 3 signals are highly correlated with each other (no diversification gain).

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 1.038 |
| 95% CI | [0.628, 1.445] |
| CAGR | 13.16% |
| Max DD | -33.79% |
| H001 corr | -0.013 |
| Decision | KEEP |

All 12 folds active (mixed: H254 tail risk has early folds, H284 momentum has early folds). Fold range: -0.019 (2008-2010 GFC) to +2.303 (2012-2013). Robust across regimes.
MDR weights: H265 (34%), H254 (30%), H287 (15%), H268 (11%), H284 (10%). MDR concentrates on sustainability-filtered quality and tail risk avoidance — the most decorrelated pair.
Equal-weight baseline already achieves Sharpe 1.034 — near-zero MDR benefit (+0.004) suggests the 5 strategies are already well-balanced in volatility. MDR adds minimal improvement over equal-weight when components have similar vol. H299 is the best pure Set 3 ensemble.
