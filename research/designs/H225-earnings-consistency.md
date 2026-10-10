# H225 — EarningsConsistency

## Hypothesis

Multi-period earnings consistency: z(1yr NI growth) + z(2yr NI growth). By averaging 1-year and 2-year earnings growth z-scores, we distinguish consistent multi-period growers from companies with a single good year or a mean-reverting spike. A company with strong NI growth at both the 1-year and 2-year horizon has demonstrated sustained profitability improvement, not just a one-off beat.

The hypothesis tests whether earnings momentum has a "persistence" quality: consistent growers are not just trending but have structural, sustainable profitability improvements that the market underreacts to (consistent with the PEAD literature extended to multi-period horizons).

**Mechanics**: ni_growth_1yr = (NI - NI.shift(252)) / |NI.shift(252)|, clipped [-1, 3]. Same for 2yr using shift(504). score = z(g_1yr) + z(g_2yr). Sector-neutral top 20%. Monthly rebalance.

## Expected behavior by regime

- **2000-2002 tech crash**: Two-year growth looks back to 1998-2000 peak NI (often negative). Signal is noisy in crisis; zero for EDGAR-uncovered period anyway.
- **2008-2009 GFC**: Post-2010 folds see companies with both 1-year and 2-year NI recovery scoring well.
- **2020 COVID**: The 2-year window captures COVID disruption; folds 10-11 see recovery compound with prior earnings trend.
- **Low-vol 2013-2017**: Steady compounders show consistent growth at both horizons — ideal regime.

## Success criteria

- OOS Sharpe > 0.51 with CI lower bound > 0.
- H001 corr near zero (pure fundamental signal).
- Folds 6-11 above 1.0 average (multi-period consistency should be a robust signal in the covered window).

## Falsification

If folds 6-11 average is weak (<0.7), multi-period earnings momentum is not more predictive than single-period. If CI lower ≤ 0, the composite adds no signal over simpler single-year approaches.

## Results (2026-05-25)

**Decision: KEEP** (weakest Sharpe in batch, 6 zero folds)

| Metric | Value |
|---|---|
| OOS Sharpe | 0.692 |
| 95% CI | [0.288, 1.104] |
| CAGR | 7.73% |
| Max DD | -37.42% |
| H001 corr | -0.017 |

**Per-fold Sharpes**: 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.198, 1.887, 0.795, 1.685, 0.968, 0.985

Folds 0-5 exactly zero (EDGAR gap). Folds 6-11 average 1.086 — consistent, but fold 6 (0.198) is weak, suggesting the 2-year lookback hurts early in the EDGAR coverage period (limited data). Folds 7-11 average 1.264, which is strong once 2-year data is fully populated. Lowest Sharpe in batch (0.692) but still above the kill bar. Max DD -37.42% is strong. The 756-day sensitivity variant shows improvement (0.740) — longer lookbacks may be better for consistency.
