# H015 — Dual MA Long/Short (20/100): Stats

**Status:** killed
**Family:** trend / long-short
**Instrument:** SPY
**Run date:** 2026-05-21

## Parameters

| Parameter | Value |
|-----------|-------|
| Fast MA | 20 days |
| Slow MA | 100 days |
| Direction | long/short (±1) |

## OOS metrics (combined across 12 folds)

| Metric | H015 | SPY benchmark | Delta |
|--------|------|---------------|-------|
| Sharpe | 0.08 | 0.51 | −0.43 |
| 95% CI on Sharpe | [−0.33, 0.50] | — | — |
| CAGR | −0.28% | 8.35% | −8.63 pp |
| Max drawdown | −51.00% | −55.92% | −4.92 pp |
| Trades | 71 | — | — |

## Sensitivity (fast/slow pair)

| Params | Sharpe |
|--------|--------|
| MA(10,50) | 0.03 |
| MA(20,100) canonical | 0.08 |
| MA(20,50) | 0.00 |

## Per-fold OOS Sharpe

`['0.14', '-0.72', '0.52', '-0.76', '0.16', '0.56', '-0.45', '0.93', '-0.45', '0.87', '0.10', '0.09']`

## Correlation with H001: +0.213

## Decision: killed

Near-zero Sharpe (0.08), negative CAGR (−0.28%), and negative CI lower bound (−0.33) make this a clear kill. The 20/100 MA pair crossovers too frequently in the SPY data, generating whipsaw losses on both the long and short side that overwhelm any trend signal. Per-fold Sharpes are highly variable (range: −0.76 to +0.93), indicating the strategy is reacting to noise rather than a stable signal.

Notable: H015 has the lowest H001 correlation of any strategy in batch 2 (+0.213). The faster MA pair genuinely captures different dynamics. However, those dynamics appear to be largely noise at the daily bar frequency used here. A faster-frequency dataset (hourly bars) might make this more viable, but at daily resolution the 20/100 crossing is not information.
