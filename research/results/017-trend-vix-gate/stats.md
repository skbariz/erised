# H017 — Trend + VIX Gate Long/Short: Stats

**Status:** killed
**Family:** trend / hybrid
**Instrument:** SPY + VIX
**Run date:** 2026-05-21

## Parameters

| Parameter | Value |
|-----------|-------|
| Fast MA | 50 days |
| Slow MA | 200 days |
| VIX extreme gate | > 40 (forced flat) |
| Direction | long/short (±1) when active |

## OOS metrics (combined across 12 folds)

| Metric | H017 | SPY benchmark | Delta |
|--------|------|---------------|-------|
| Sharpe | 0.43 | 0.51 | −0.08 |
| 95% CI on Sharpe | [0.03, 0.83] | — | — |
| CAGR | 5.92% | 8.35% | −2.43 pp |
| Max drawdown | −39.10% | −55.92% | −16.82 pp |
| Trades | 85 | — | — |

## Sensitivity (vix_extreme threshold)

| vix_extreme | Sharpe |
|-------------|--------|
| 30 | 0.59 |
| **40** | **0.43** |
| 50 | 0.41 |

## Per-fold OOS Sharpe

`['-0.18', '0.51', '0.93', '0.57', '0.83', '0.79', '-0.71', '1.51', '-0.02', '1.13', '0.24', '0.42']`

## Correlation with H001: +0.701

## Decision: killed

Sharpe 0.43 is below SPY benchmark (0.51), failing the primary keep criterion.

The hypothesis that VIX > 40 identifies whipsaw-prone crash environments holds partially — but the VIX=40 gate triggers too rarely to materially change performance, and when it does force flat, the strategy misses both the short losses AND the occasional short gains from correctly-called bear moves.

The long/short structure (same issue as H013/H014) is the primary problem. Adding the VIX gate simply reduces the number of short periods without fixing the fundamental issue that SPY short positions during the 2000–2022 period are net-negative on this timescale.

**Sensitivity note:** vix_extreme=30 yields Sharpe 0.59, which would have passed the keep criteria. However, a lower VIX threshold means more time in cash, effectively making H017 more similar to a long-only strategy gated by VIX. This is not the hypothesis as specified, and cherry-picking the threshold post-hoc is curve-fitting.
