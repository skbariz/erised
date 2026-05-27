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
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
