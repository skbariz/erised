# H006 — VIX-Filtered Mean Reversion: Stats

**Status:** killed  **Family:** mean reversion  **Run date:** 2026-05-21

## Canonical parameters: lookback=20, threshold=2.0, hold_days=2, vix_threshold=20

| Metric | H006 | SPY | Delta |
|--------|------|-----|-------|
| Sharpe | 0.01 | 0.51 | -0.50 |
| 95% CI | [-0.39, 0.43] | — | — |
| CAGR | -0.14% | 8.35% | — |
| Max DD | -22.60% | -55.92% | — |
| Trades | 230 | — | — |

## 1-D sensitivity (vix_threshold)

| vix_threshold | Sharpe |
|---------------|--------|
| 15 | -0.00 |
| **20** | **0.01** |
| 25 | 0.09 |

## Per-fold Sharpe
['−0.09', '−1.06', '0.91', '0.00', '−0.63', '0.87', '−0.10', '0.61', '−0.44', '−0.00', '0.95', '−0.01']

## Correlation with H001: +0.402

## Kill reason
The VIX filter does not improve the mean-reversion signal — in fact it appears slightly harmful. This contradicts the hypothesis that panic regimes are where reversion works. The unfiltered short-hold reversion (H003) also has near-zero Sharpe, which confirms the 2-day hold is fundamentally weak regardless of VIX filtering.
