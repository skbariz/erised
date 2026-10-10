# H243: GrossProfitGrowth — YoY Absolute GP Growth / Total Assets

## Hypothesis

Growing the absolute gross profit pool (not just the ratio) signals genuine business expansion. A company that increases its total gross profit relative to its asset base is simultaneously growing revenues and maintaining or expanding margins — a compounding quality signal. Distinct from GP/TA level (H227: static quality) and GP/revenue level (H239: pricing power).

Signal: (GP_t − GP_{t−1y}) / TA_{t−1y}, clipped [-1.0, 2.0]. Sector-neutral top 20%, monthly rebalance.

Expected regime behavior:
- **2000–2002 tech crash**: GP-growing companies have real businesses; avoids revenue-only growers.
- **2008–2009 GFC**: GP contraction is expected; signal selects relative outperformers.
- **2020 COVID**: Sharp GP divergence by sector; sector-neutral design is critical.
- **2013–2017 low-vol**: Steady growth period; GP growth should be systematically rewarded.

Success criteria: OOS Sharpe > 0.51, CI lower bound > 0. Expect flat window sensitivity (GP growth is slow-moving).

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.791 |
| 95% CI | [0.381, 1.202] |
| CAGR | 9.47% |
| Max DD | −34.90% |
| H001 corr | −0.015 |
| Decision | **KEEP** |

**Sensitivity (window):** 126d → 0.813 | 252d → 0.791 | 504d → 0.814. Remarkably flat — signal is robust across all lookback windows; 126d and 504d slightly stronger than canonical 252d.

**Folds 6–11 Sharpe:** 0.350, 1.919, 0.985, 1.945, 1.265, 1.121. Folds 0–5 zero.

**Batch star** — best Sharpe (0.791) and best MaxDD (−34.90%) in Batch 9. Flat window profile confirms signal robustness. GP growth / TA is a novel formulation that captures both growth AND asset efficiency in one metric.
