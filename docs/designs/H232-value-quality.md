# H232 — ValueQuality

## Hypothesis

Value-quality composite: z(book_equity/price) + z(GP/TA). Asness, Frazzini, Pedersen (2019) "Quality Minus Junk" (QMJ) shows that cheap AND high-quality stocks outperform expensive AND low-quality stocks. This is a simplified version combining B/M value with gross profitability quality. No price momentum — pure fundamental composite.

## Results (2026-05-25)

**Decision: KEEP**

| Metric | Value |
|---|---|
| OOS Sharpe | 0.864 |
| 95% CI | [0.451, 1.274] |
| CAGR | 11.00% |
| Max DD | -35.61% |
| H001 corr | -0.019 |

**Per-fold Sharpes**: 0.000, 0.000, 0.000, 0.000, 0.000, 0.475, 0.824, 2.054, 0.864, 2.183, 1.145, 1.100

At 10% quantile: Sharpe 0.871, Max DD -32.40% — the tightest quality+value companies have the lowest drawdown in Batch 7. Fold 9 (2.183) is the highest in the batch. H001 corr -0.019 confirms pure fundamental composite is decorrelated from trend-following.
