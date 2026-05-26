# H237: DebtCoverage — OCF / Long-Term Debt Solvency Ratio

## Hypothesis

Companies that can cover their long-term debt many times over with operating cash flow have superior long-term financial health. OCF/LTD measures how many times over a company could pay off its long-term obligations from a single year of cash generation. High ratio = structural solvency, low credit risk, and access to cheap capital when needed.

Expected regime behavior:
- **2000–2002 tech crash**: High OCF/LTD firms are insulated from credit market tightening.
- **2008–2009 GFC**: Strong signal — over-leveraged firms get punished; cash-generative firms survive.
- **2020 COVID**: Strong signal — zero-LTD firms and high-coverage firms retain access to markets.
- **2013–2017 low-vol**: Modest — in a benign credit environment, solvency premium compresses.

Success criteria: OOS Sharpe > 0.51, CI lower bound > 0, H001 corr < 0.10.
Falsification: If solvency ratio has no cross-sectional predictive power, folds 6-11 would be near zero.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.729 |
| 95% CI | [0.329, 1.139] |
| CAGR | 9.04% |
| Max DD | −36.18% |
| H001 corr | −0.020 |
| Decision | **KEEP** |

**Sensitivity (within_sector_quantile):** 10% → 0.729, 20% → 0.729, 30% → 0.725. Extremely flat — signal is robust.

**Folds 6–11 Sharpe:** 0.778, 2.060, 0.701, 1.779, 0.937, 1.012. Folds 0–4 zero; fold 5 shows 0.116 (partial EDGAR coverage).

Zero-debt companies treated as NaN (rather than infinite coverage) to avoid population distortion. Clipping at [-2, 10] handles outliers. Consistent with the broader solvency factor family — all have H001 corr near -0.020 and folds 0-5 zero.
