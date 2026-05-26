# H220 — ResidualMomQuality

## Hypothesis

Gate H163 residual momentum by balance-sheet accruals quality. High accruals (large gap between accounting earnings and cash earnings) is a well-documented predictor of future stock underperformance (Sloan 1996). By excluding the top 20% accruals stocks from the residual momentum universe, we remove the "low quality" winners that momentum would otherwise select — stocks whose price appreciation is supported by aggressive accounting rather than real cash generation.

**Mechanics**: (1) Compute CAPM-residual momentum (same as H163). (2) Compute balance-sheet accruals = (NWC_change) / avg_total_assets. (3) Rank stocks by accruals. (4) Zero weight for top 20% accruals (worst quality). (5) Sector-neutral top 20% of remaining universe.

## Expected behavior by regime

- **2000-2002 tech crash**: Tech companies had high accruals (revenue recognition, capitalized costs) → excluded by quality gate → strategy avoids the crash.
- **2008-2009 GFC**: Financial stocks with inflated balance sheets (high accruals) → excluded. Non-financial residual momentum survivors selected.
- **2020 COVID**: Accruals less predictive in short sharp crashes; quality gate neutral.
- **Low-vol 2013-2017**: Quality filter selects more conservative compounders with real cash generation.

## Success criteria

- OOS Sharpe > 0.51 with CI lower bound > 0.
- H001 corr near zero (inherits from H163 structure).
- Sharpe ≥ H163 baseline (the quality gate should improve or maintain performance by removing low-quality momentum stocks).

## Falsification

If Sharpe falls meaningfully below H163, the accruals gate removes too many stocks from the momentum universe and introduces noise. If CI lower ≤ 0, the quality-gating adds no incremental signal beyond momentum alone.

## Results (2026-05-25)

**Decision: KEEP**

| Metric | Value |
|---|---|
| OOS Sharpe | 0.958 |
| 95% CI | [0.558, 1.369] |
| CAGR | 19.11% |
| Max DD | -51.94% |
| H001 corr | +0.001 |

**Per-fold Sharpes**: 1.599, 1.015, 1.231, 1.597, 1.229, -0.120, 0.785, 2.328, 0.966, 1.599, 1.012, 0.840

Near-zero H001 correlation (+0.001). Strong and consistent fold performance (11/12 positive). The accruals quality gate modestly reduces Sharpe vs H216 (0.958 vs 0.983) and maintains similar Max DD, suggesting the quality filter is slightly too restrictive rather than clearly additive. However, the CI is tight and the overall result is robust. Valuable as a portfolio component for its near-zero H001 correlation.
