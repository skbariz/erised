# H003 — Short-Hold Mean Reversion (2-day): Stats

**Status:** killed  **Family:** mean reversion  **Run date:** 2026-05-21

## Canonical parameters: lookback=20, threshold=2.0, hold_days=2

| Metric | H003 | SPY | Delta |
|--------|------|-----|-------|
| Sharpe | 0.01 | 0.51 | -0.50 |
| 95% CI | [-0.39, 0.42] | — | — |
| CAGR | -0.21% | 8.35% | -8.56 pp |
| Max DD | -26.84% | -55.92% | — |
| Trades | 446 | — | — |

## 1-D sensitivity (hold_days)

| hold_days | Sharpe |
|-----------|--------|
| 1 | 0.45 |
| **2** | **0.01** |
| 3 | 0.06 |

## Per-fold Sharpe
['−0.09', '−0.97', '0.60', '−0.42', '−0.57', '0.77', '0.00', '0.94', '−0.19', '0.56', '0.08', '0.05']

## Correlation with H001: +0.440

## Kill reason
CI contains zero ([-0.39, 0.42]). Essentially a flat signal. The 1-day hold variant (Sharpe 0.45) is the best mean-reversion configuration seen so far and may be worth a dedicated hypothesis.
