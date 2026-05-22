# H007 — 12-1 Month Time-Series Momentum: Stats

**Status:** kept  **Family:** trend / momentum  **Run date:** 2026-05-21

## Canonical parameters: long_months=12, skip_months=1

| Metric | H007 | SPY | Delta |
|--------|------|-----|-------|
| Sharpe | 0.62 | 0.51 | +0.11 |
| 95% CI | [0.21, 1.01] | — | — |
| CAGR | 8.48% | 8.35% | +0.13 pp |
| Max DD | -33.72% | -55.92% | -22.20 pp |
| Trades | 97 | — | — |

## 1-D sensitivity (long_months, skip=1)

| long_months | Sharpe |
|-------------|--------|
| 6 | 0.51 |
| 9 | 0.62 |
| **12** | **0.62** |

## Per-fold Sharpe
['0.93', '0.05', '1.06', '0.77', '0.33', '0.14', '0.49', '1.49', '0.40', '2.08', '0.37', '0.56']

## Correlation with H001: +0.892

## Decision: kept — strong standalone signal; does not add ensemble diversification over H001

The Sharpe (0.62) and CI ([0.21, 1.01]) match H001 almost exactly, confirming that 12-month momentum and MA(50,200) are measuring the same underlying trend signal via different lookback windows. The +0.892 returns correlation confirms this. Both are effective trend-following strategies; H007 adds no ensemble value over H001 but validates the trend signal family's robustness.
