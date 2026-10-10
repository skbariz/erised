# H001 — MA(50,200) Crossover: Stats

**Status:** kept (baseline)
**Family:** trend
**Instrument:** SPY
**Run date:** 2026-05-21

## Parameters

| Parameter | Value |
|-----------|-------|
| Fast MA | 50 days |
| Slow MA | 200 days |
| Direction | long-only |
| Sizing | binary (0 / +1) |

## Walk-forward configuration

| Setting | Value |
|---------|-------|
| Research period | 1993-01-29 to 2022-12-30 |
| Train window | 5 years |
| Test window | 2 years |
| Folds | 12 |
| OOS periods | 6,036 bars |
| Commission | 0.5 bps |
| Slippage | 1.0 bps |

## OOS metrics (combined across 12 folds)

| Metric | H001 | SPY benchmark | Delta |
|--------|------|---------------|-------|
| Sharpe | 0.61 | 0.51 | +0.10 |
| 95% CI on Sharpe | [0.18, 1.02] | — | — |
| CAGR | 7.86% | 8.35% | −0.49 pp |
| Ann vol | 13.91% | 19.66% | −5.75 pp |
| Max drawdown | −33.72% | −55.92% | −22.20 pp |
| Calmar | 0.23 | 0.15 | +0.08 |
| Sortino | n/a | n/a | — |
| Hit rate | 55.20% | 54.54% | +0.66 pp |
| Total trades | 25 | — | |
| Turnover / day | 0.40% | — | |

## In-sample reference (full 1993–2022, 7,536 periods)

| Metric | H001 | SPY |
|--------|------|-----|
| Sharpe | 0.71 | 0.58 |
| Max DD | −33.72% | −55.19% |

IS slightly better than OOS — consistent with mild conventional fit (50/200 are
canonical numbers, not grid-searched in our hands).

## Ensemble status

Included as the trend-following leg. Baseline for all future correlation checks.

## Per-fold OOS Sharpe

See `research_log/001-ma-50-200-baseline.md` for full per-fold breakdown.
