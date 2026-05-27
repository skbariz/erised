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
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
