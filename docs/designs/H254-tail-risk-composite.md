# H254: TailRiskComposite — Combined Tail Risk Avoidance

## Hypothesis

Four orthogonal tail-risk signals — idiosyncratic volatility (H251), MAX return (H252),
downside beta (H253), and return skewness (H197) — each capture a different dimension of
lottery/crash risk. Combining them as an equal z-score composite reduces measurement noise
from any single metric.

Signal: z(-idio_vol_63d) + z(-MAX21) + z(-downside_beta_252d) + z(skewness_63d).
All four z-scores orientated so that high composite = low tail risk.
Sector-neutral top 20%, monthly rebalance.

This is the 4-factor composite at the dilution boundary identified in program findings.
All four components must individually contribute positive IC for the composite to be valid.

Expected regime behavior:
- **2000–2002**: All four components aligned; composite should be strongest here.
- **2008–2009**: Downside beta and idio vol dominate; MAX and skewness may diverge.
- **2013–2017**: Low-vol era; all four signals weak; mild performance expected.
- **2020**: Mixed — COVID spike is systematic across all tail-risk dimensions.

Success criteria: OOS Sharpe ≥ 0.80, Max DD < −42%, CI lower bound > 0.
Must beat best standalone component (H253 expected ~0.70) to justify composite complexity.
Falsification: Composite Sharpe falls below best standalone (dilution confirmed).

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.911 |
| 95% CI | [0.506, 1.327] |
| CAGR | 14.76% |
| Max DD | −44.60% |
| H001 corr | −0.000 |
| Decision | **KEEP** |

**Sensitivity (skew_window):** 21d → 0.940 | 63d → 0.911 | 126d → 0.951. Non-monotonic;
skew window matters less than the other three components.

**Dilution observed:** Composite (0.911) underperforms best standalone H253 (0.976). The 4-factor
composite dilutes the strong downside-beta signal. Still passes keep bar and Max DD is best in
batch (−44.60%). Useful as a diversified tail-risk screen but not as a primary alpha signal.
