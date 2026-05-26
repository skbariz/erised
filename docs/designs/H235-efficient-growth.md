# H235 — EfficientGrowth

## Hypothesis

Efficient growth: z(YoY revenue growth) + z(OCF/TA). Revenue growth backed by operating cash flow generation per asset dollar. Filters out "growth at any cost" companies — if a company is growing revenue but has low OCF/TA, it is burning cash to fund growth. High OCF/TA + high revenue growth = genuinely efficient, sustainable expansion.

## Results (2026-05-25)

**Decision: KEEP**

| Metric | Value |
|---|---|
| OOS Sharpe | 0.766 |
| 95% CI | [0.362, 1.170] |
| CAGR | 8.79% |
| Max DD | -34.51% |
| H001 corr | -0.022 |

**Per-fold Sharpes**: 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.464, 2.028, 0.789, 1.660, 0.946, 1.284

Folds 0-5 zero (EDGAR gap). Fold 11 (1.284) is particularly strong — suggests efficient growth signal became more relevant in the 2019-2022 period. 252d window is optimal. Max DD -34.51% is strong. H001 corr -0.022 (most negative in Batch 7) — OCF/TA quality anchor adds decorrelation vs. H228 pure revenue growth (-0.023).
