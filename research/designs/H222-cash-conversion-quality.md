# H222 — CashConversionQuality

## Hypothesis

Cash conversion efficiency: OCF / gross_profit. For every dollar of gross profit, what fraction becomes operating cash? High values indicate that accounting profits translate into real cash flows — a hallmark of high-quality earnings that cannot be sustained by aggressive revenue recognition or cost deferral. This is related to but distinct from Sloan (1996) accruals: accruals measures NI vs OCF, this measures OCF vs gross_profit (the upstream measure before operating expenses).

The economic intuition is that companies with high OCF/GP ratios have efficient working capital management, genuine pricing power, and conservative accounting — all of which predict future outperformance through both fundamental quality and lower risk of earnings restatements.

**Mechanics**: cash_conv = OCF / gross_profit (filtered to GP > 0). Sector-neutral top 20% by cash conversion ratio. Monthly rebalance. No window parameter — uses current-state ratio, not change.

## Expected behavior by regime

- **2000-2002 tech crash**: Tech companies burned cash despite high gross profits — low cash conversion → excluded. Strategy avoids the worst crash stocks.
- **2008-2009 GFC**: Financials/real estate had OCF collapse; non-financial high-conversion companies selected.
- **2020 COVID**: Consumer staples and healthcare (high OCF/GP) selected — defensive quality.
- **Low-vol 2013-2017**: Quality companies with high OCF/GP tend to be steady compounders — good regime.

## Success criteria

- OOS Sharpe > 0.51 with CI lower bound > 0.
- H001 corr near zero (pure fundamental quality signal).
- Max DD substantially better than SPY (quality companies have lower bankruptcy risk).

## Falsification

If CI lower ≤ 0 or Sharpe < 0.51, the OCF/GP ratio does not distinguish future winners from losers. If Max DD approaches SPY (-55.92%), quality selection doesn't provide downside protection.

## Results (2026-05-25)

**Decision: KEEP** — Batch 5 star

| Metric | Value |
|---|---|
| OOS Sharpe | 0.829 |
| 95% CI | [0.423, 1.240] |
| CAGR | 10.24% |
| Max DD | -35.83% |
| H001 corr | -0.020 |

**Per-fold Sharpes**: 0.000, 0.000, 0.000, 0.000, 0.000, 0.602, 0.898, 2.176, 0.854, 1.928, 0.993, 0.942

Fold 5 (2008-2010) active with Sharpe 0.602 — OCF/GP data was partially available before 2010 (Q3/Q4 XBRL filers). Folds 6-11 average 1.299, highest in batch. Best Sharpe (0.829) and best Max DD (-35.83%) in Batch 5. H001 corr -0.020 confirms decorrelation. The level-based (not change-based) approach is more robust, providing a stable signal.
