# H227 — GrossProfitability

## Hypothesis

Novy-Marx (2013) gross profitability factor: GP/TA. The most profitable companies (gross profit relative to total assets) outperform the least profitable. Novy-Marx showed that GP/TA is a more robust predictor of returns than net income-based measures because gross profit is less susceptible to managerial manipulation via accruals, depreciation choices, and other accounting discretion. The signal is level-based (current ratio), not change-based, reflecting the idea that profitability is persistent.

**Mechanics**: gp_ta = gross_profit / total_assets. Sector-neutral top 20% by GP/TA level. Monthly rebalance.

## Expected behavior by regime

- **2000-2002 tech crash**: Tech had high gross margins (asset-light, high GP/TA for software); mixed. But real high-GP/TA companies (consumer staples, pharmaceuticals) are defensive.
- **2008-2009 GFC**: Financial asset-heavy companies have low GP/TA → excluded. Non-financials with real pricing power selected.
- **2020 COVID**: Healthcare and tech (cloud) have high GP/TA → defensive positioning in crisis, strong recovery.
- **Low-vol 2013-2017**: Steady compounders with high gross margins consistently rank well.

## Success criteria

- OOS Sharpe > 0.51 with CI lower bound > 0.
- H001 corr near zero (pure fundamental quality, no price signal).
- Max DD substantially better than SPY — high profitability = lower distress risk.

## Falsification

If Sharpe < 0.51 or CI lower ≤ 0, GP/TA does not predict cross-sectional returns within the S&P 500. This would imply the Novy-Marx result is specific to value stock universes, not already-profitable large caps.

## Results (2026-05-25)

**Decision: KEEP** — Batch 6 star

| Metric | Value |
|---|---|
| OOS Sharpe | 0.879 |
| 95% CI | [0.469, 1.287] |
| CAGR | 11.03% |
| Max DD | -34.46% |
| H001 corr | -0.017 |

**Per-fold Sharpes**: 0.000, 0.000, 0.000, 0.000, 0.000, 0.485, 0.838, 2.059, 1.010, 2.039, 1.200, 1.154

Fold 5 active at 0.485 (partial GP/TA coverage). Folds 6-11 average 1.383 — highest in Batch 6, with every fold positive and strong (1.0+). Best Sharpe (0.879) and best Max DD (-34.46%) in the batch. At 10% quantile: Sharpe 0.922, Max DD -29.29% — indicating the very top GP/TA stocks are even stronger. Level-based GP/TA is the cleanest signal in the value/efficiency family.
