# H300: GrandCrossProgram300MDR — Terminal 300-Strategy MDR Champion

## Hypothesis

H300 is the terminal champion candidate for the entire 300-strategy research program.
MDR (Maximum Diversification Ratio) over the full program's best keepers combines:
- H095 components (H001-H100 best macro/ETF diversifiers)
- H200 components (H101-H200 best cross-sectional equity)  
- Best 5-10 kept strategies from H251-H299 (Set 3 risk-avoidance)

Each program phase found fundamentally different return drivers — macro ETF rotation,
cross-sectional fundamental factors, and risk-avoidance signals. MDR finds the weights
that maximize the ratio of portfolio diversification to portfolio volatility.

Signal: SLSQP MDR optimization over ~15-20 candidate strategies drawn from:
- All H095 weights (H043, H078, H035, H060, H025)
- Top H200 components by MDR contribution
- Top H251-H299 kept strategies (cross-program diversifiers)

Target: Beat H200 (Sharpe 1.350, MaxDD −8.83%) — the current program champion.

Success criteria: OOS Sharpe ≥ 1.40, MaxDD < −8.0%, CI lower > 0.
Falsification: Adding Set 3 equity strategies to the macro-dominated H200 ensemble
reduces Sharpe — Set 3 signals correlate with H200's cross-sectional equity components.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 1.525 |
| 95% CI | [1.134, 1.921] |
| CAGR | 5.82% |
| Max DD | -6.86% |
| H001 corr | +0.652 |
| Decision | KEEP |

All 12 folds active. Fold range: +0.791 (2018-2020) to +2.471 (2016-2018). No negative folds.
NEW PROGRAM CHAMPION: Sharpe 1.525 / MaxDD -6.86% — beats H200 (1.350/-8.83%).
MDR top weights: H194 LongShortReversal (23%), H016 VolScaledTrend (11%), H043 SellInMayBondConfirm (10%), H068 VrpCalendarHybrid (9%), H076 FourAssetRiskParity (8%), H265 SustainabilityFilteredProfitability (7%).
Set 3 contribution: H265 (7%), H268 (1%), H287 (1%) — MDR correctly identifies Set 3 as incremental to the established H200 ensemble. The dominant diversifiers remain H194 (L/S reversal) and macro strategies.
H001 corr +0.652 due to H001-H100 macro strategies dominating the portfolio via MDR concentration.
