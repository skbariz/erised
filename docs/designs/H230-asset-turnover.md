# H230 — AssetTurnover

## Hypothesis

YoY improvement in asset turnover (revenue / total_assets). Asset turnover is the efficiency component of the DuPont decomposition of ROE. Companies that are generating more revenue from the same asset base are becoming more operationally efficient — reducing asset bloat, improving inventory management, or finding higher-utilization uses for existing assets. This is distinct from revenue growth (H228, which measures absolute growth) and operating leverage (H229, which measures margin expansion).

**Mechanics**: at = rev / total_assets. at_change = at - at.shift(252). Sector-neutral top 20% by at_change. Monthly rebalance.

## Expected behavior by regime

- **2000-2002 tech crash**: Capital-intensive tech companies (expensive infrastructure, low asset utilization) had declining asset turnover. Asset-light companies excelled.
- **2008-2009 GFC**: Post-crisis, companies that reduced working capital and assets fastest while maintaining revenue had the best asset turnover improvement.
- **2020 COVID**: E-commerce fulfillment efficiency vs. traditional retail collapse. Strong signal for operational transformation.
- **Low-vol 2013-2017**: Industrial and technology companies with steady operational improvement.

## Success criteria

- OOS Sharpe > 0.51 with CI lower bound > 0.
- H001 corr near zero (pure fundamental efficiency metric).
- Distinct from H228 (revenue growth) and H229 (operating leverage) — different dimension of DuPont.

## Falsification

If Sharpe < 0.51 or CI lower ≤ 0, operational efficiency improvement does not predict cross-sectional returns. The signal might be dominated by sector effects (service companies have inherently higher AT than capital-intensive industrials).

## Results (2026-05-25)

**Decision: KEEP**

| Metric | Value |
|---|---|
| OOS Sharpe | 0.741 |
| 95% CI | [0.334, 1.149] |
| CAGR | 8.52% |
| Max DD | -36.32% |
| H001 corr | -0.021 |

**Per-fold Sharpes**: 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.618, 2.043, 0.743, 1.690, 0.934, 1.036

Folds 0-5 zero (EDGAR gap). Folds 6-11 average 1.177. The 126d window (0.748) and 252d window (0.741) are similar — AT changes are measured on quarterly filing cycles so medium-term lookbacks are comparable. Longer windows (504d) degrade (0.659), suggesting asset turnover improvements are shorter-cycle signals. H001 corr -0.021, consistent with the quality factors cluster.
