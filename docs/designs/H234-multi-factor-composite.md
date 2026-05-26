# H234 — MultiFactorComposite

## Hypothesis

4-factor equal-weight composite: z(12-1m momentum) + z(GP/TA) + z(-BS_accruals) + z(-asset_growth). Momentum + quality + accruals quality + capital discipline. Tests whether a broader factor composite consistently outperforms any single factor. Each component is academically established; the combination reduces idiosyncratic risk of any single signal.

## Results (2026-05-25)

**Decision: KEEP**

| Metric | Value |
|---|---|
| OOS Sharpe | 0.857 |
| 95% CI | [0.442, 1.266] |
| CAGR | 10.48% |
| Max DD | -35.76% |
| H001 corr | -0.013 |

**Per-fold Sharpes**: 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.748, 2.251, 0.795, 2.047, 1.200, 1.033

Folds 6-11 average 1.346. The momentum component provides Fold 0-5 coverage potential but EDGAR gap zeros out accruals/asset_growth, so the 4-factor composite falls back to momentum+zero. H001 corr -0.013 (the 4 factors partially cancel the momentum's mild positive H001 corr). Quantile sensitivity virtually flat (0.870/0.857/0.842) — very stable.
