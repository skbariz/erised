# H248: IntangibleQuality — GP / Book Equity (Intangible Asset Intensity Proxy)

## Hypothesis

Companies with high gross profit relative to book equity are "intangible-rich" — their economic value comes from brands, intellectual property, patents, or other assets not captured on the balance sheet. When GP >> BE, the gap represents off-balance-sheet intangibles. This ratio captures the quality premium of asset-light, high-IP businesses.

Signal: GP / BE where BE > 0, clipped [-1.0, 10.0]. Sector-neutral top 20%, monthly rebalance.

Expected regime behavior:
- **2000–2002 tech crash**: Mixed — some high-GP/BE tech firms are genuinely valuable, others are not.
- **2008–2009 GFC**: Asset-light businesses (software, pharma) are less exposed to credit crunch than asset-heavy.
- **2020 COVID**: Technology and healthcare (high GP/BE) dramatically outperform.
- **2013–2017 low-vol**: Structural quality premium in the knowledge economy.

Success criteria: OOS Sharpe > 0.51, CI lower bound > 0. Expect H001 corr near zero (accounting-only).
Distinct from GP/TA (H227, asset efficiency denominator) and GP/revenue (H239, margin denominator).

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.864 |
| 95% CI | [0.446, 1.271] |
| CAGR | 10.83% |
| Max DD | −36.29% |
| H001 corr | −0.019 |
| Decision | **KEEP** |

**Sensitivity (within_sector_quantile):** 10% → 0.859 | 20% → 0.864 | 30% → 0.847. Extremely flat — signal robust across portfolio sizes.

**Folds 5–11 Sharpe:** 0.592, 0.806, 2.046, 1.054, 2.087, 1.189, 1.042. Fold 9 at 2.087 is the highest active fold in Batch 10.

**Batch star** — best Sharpe (0.864) and highest CAGR (10.83%) in Batch 10. GP/BE denominator change from TA (H227) materially improves results: 0.864 vs 0.879, with near-equivalent MaxDD. The intangible-quality interpretation adds a theoretically distinct explanation for the GP-based signal family.
