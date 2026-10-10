# H217 — ResidualMomVolScaled

## Hypothesis

Apply idiosyncratic vol-target scaling to H163 residual momentum — the same mechanism that took H021 (risk parity) to H086 (LowVolTargetPortfolio, Sharpe 1.109). H163 selects stocks by CAPM-residual momentum; this strategy additionally scales each position weight by `min(1, target_vol / realized_idiosyncratic_vol)`. The intuition: high idiosyncratic vol stocks that appear in the top momentum quantile are penalized in size, reducing the risk contribution of volatile winners and smoothing the overall portfolio volatility.

**Mechanics**: (1) Compute CAPM residual (same as H163). (2) Compute 60-day rolling std of those residuals, annualized. (3) Weight *= min(1, 12% / resid_vol). (4) Sector-neutral top 20% baseline, then apply scalar.

## Expected behavior by regime

- **2000-2002 tech crash**: Tech stocks have high idiosyncratic vol — scalars shrink their weights, limiting concentration just when crashes hit.
- **2008-2009 GFC**: Idiosyncratic vol spikes system-wide; most scalars < 1, net reducing portfolio exposure — natural de-risking.
- **2020 COVID**: Same mechanism — sharp vol spike automatically reduces exposure at the right moment.
- **Low-vol 2013-2017**: Scalars near 1.0 → effectively identical to H163.

## Success criteria

- OOS Sharpe > 0.51 with CI lower bound > 0.
- H001 corr < 0.10 (inherits from H163 structure).
- Max DD improvement over H163 (the primary motivation for vol scaling).

## Falsification

If Max DD is not improved over H163, idiosyncratic vol scaling does not help cross-sectional strategies (unlike the multi-asset case). If Sharpe falls substantially below H163, the vol-penalization is distorting the signal too much.

## Results (2026-05-25)

**Decision: KEEP**

| Metric | Value |
|---|---|
| OOS Sharpe | 0.936 |
| 95% CI | [0.532, 1.353] |
| CAGR | 17.76% |
| Max DD | -51.41% |
| H001 corr | +0.001 |

**Per-fold Sharpes**: 1.401, 1.004, 1.181, 1.480, 1.267, -0.148, 0.795, 2.211, 1.041, 1.706, 1.094, 0.945

Near-zero H001 correlation (+0.001). Vol scaling modestly reduces Sharpe vs H216 (0.936 vs 0.983) without significantly improving Max DD (cross-sectional strategies already hold ~100 stocks so diversification limits idiosyncratic vol impact). The vol-scaling mechanism that works powerfully on 2-3 asset portfolios (H086) is less impactful on diversified cross-sectional portfolios. Still a solid KEEP for ensemble use.
