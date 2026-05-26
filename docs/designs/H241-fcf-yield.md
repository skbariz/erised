# H241: FCFYield — OCF / Market Cap

## Hypothesis

Free cash flow yield (OCF / market_cap) bridges accounting and market pricing: a company with high operating cash flow relative to its market capitalization is "cheap on cash flow." This is the cash-flow analog to earnings yield and should predict positive abnormal returns as markets reprice the cash generation gap.

Signal: OCF / (price × shares_outstanding), clipped [-0.5, 0.5]. Sector-neutral top 20%, monthly rebalance.

Expected regime behavior:
- **2000–2002 tech crash**: Pre-revenue tech companies have deeply negative FCF yield; signal avoids them.
- **2008–2009 GFC**: Credit crunch makes real cash generation even more valuable; signal should strengthen.
- **2020 COVID**: Price collapse creates temporarily high FCF yields in beaten-down sectors.
- **2013–2017 low-vol**: Stable value factor; moderate performance expected.

Success criteria: OOS Sharpe > 0.51, CI lower bound > 0, H001 corr near zero.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.758 |
| 95% CI | [0.358, 1.166] |
| CAGR | 9.98% |
| Max DD | −42.31% |
| H001 corr | −0.021 |
| Decision | **KEEP** |

**Sensitivity (within_sector_quantile):** 10% → 0.833/MaxDD −41.11% | 20% → 0.758 | 30% → 0.764. Top 10% concentration improves Sharpe but doesn't rescue MaxDD meaningfully.

**Folds 5–11 Sharpe:** 0.248, 0.837, 2.131, 0.701, 1.834, 0.941, 0.984. Folds 0–4 zero.

MaxDD (−42.31%) is the highest in Batch 9 — likely because market cap linkage means the signal can be wrong during valuation-driven rallies. Otherwise passes cleanly.
