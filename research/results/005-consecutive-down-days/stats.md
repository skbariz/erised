# H005 — Consecutive Down Days Entry: Stats

**Status:** killed  **Family:** mean reversion  **Run date:** 2026-05-21

## Canonical parameters: n_down=3, hold_days=3

| Metric | H005 | SPY | Delta |
|--------|------|-----|-------|
| Sharpe | 0.44 | 0.51 | -0.07 |
| 95% CI | [0.03, 0.83] | — | — |
| CAGR | 4.11% | 8.35% | -4.24 pp |
| Max DD | -27.12% | -55.92% | -28.80 pp |
| Trades | 615 | — | — |

## Grid (n_down x hold_days)

| n_down | hold=2 | hold=3 | hold=5 |
|--------|--------|--------|--------|
| 2 | 0.36 | 0.40 | 0.53 |
| **3** | 0.42 | **0.44** | 0.55 |
| 4 | 0.35 | 0.34 | 0.52 |

## Per-fold Sharpe
['1.40', '0.70', '0.45', '0.39', '0.11', '−0.09', '0.65', '1.90', '−0.39', '1.52', '0.03', '0.29']

## Correlation with H001: +0.396

## Kill reason
The CI is all-positive [0.03, 0.83] — the signal is real. But returns correlation with H001 is +0.396, exceeding the 0.30 ensemble threshold. The strategy does not provide sufficient diversification to justify ensemble inclusion. The (n_down=3, hold=5) variant reaches 0.55 — worth noting for future ensemble review if correlation drops.
