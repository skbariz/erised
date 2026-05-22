# Strategy Results — Master Leaderboard

Updated after each hypothesis run. All metrics are **walk-forward out-of-sample**
(research period 1993–2022, holdout sealed and untouched). Folds: 5-year train /
2-year test, 12 non-overlapping folds. Costs: 0.5 bps commission + 1.0 bps slippage.

SPY benchmark (same OOS windows): Sharpe 0.51 | CAGR 8.35% | Max DD −55.92%

---

## Leaderboard

| # | Name | Family | Status | OOS Sharpe | 95% CI | CAGR | Max DD | H001 corr | Notes |
|---|------|--------|--------|-----------|--------|------|--------|-----------|-------|
| 001 | MA(50,200) crossover | trend | kept | 0.61 | [0.18, 1.02] | 7.86% | −33.72% | — | baseline |
| 002 | Mean reversion 2σ SPY | mean rev | killed | 0.24 | — | — | — | +0.55 | CI contains zero |
| 003 | Short-hold mean rev (2d) | mean rev | killed | 0.01 | — | — | — | — | near-zero |
| 004 | RSI oversold entry | mean rev | killed | 0.06 | — | — | — | — | CI contains zero |
| 005 | Consecutive down days | mean rev | killed | 0.44 | — | — | — | +0.40 | CI positive; fails ensemble |
| 006 | VIX-filtered reversion | mean rev | killed | 0.01 | — | — | — | — | VIX filter adds no value |
| 007 | 12-1m TS momentum | momentum | kept | 0.62 | [0.21, 1.01] | 8.48% | −33.72% | +0.892 | H001 corr too high |
| 008 | Absolute dual momentum | momentum | kept | 0.62 | [0.22, 1.02] | 8.02% | −31.23% | +0.863 | H001 corr too high |
| 009 | MACD crossover (12,26,9) | trend | killed | 0.37 | — | — | — | — | CI lower negative |
| 010 | Fast MA crossover (10/50) | trend | killed | 0.45 | — | — | — | — | dominated by H001 |
| 011 | ADX-filtered MA crossover | trend | killed | 0.27 | — | — | — | — | CI contains zero |
| 012 | Realized vol regime | vol | killed | 0.33 | — | — | — | — | CI contains zero |
| 013 | MA(50,200) long/short | trend L/S | killed | 0.36 | [−0.04, 0.77] | 5.32% | −40.16% | +0.707 | shorting hurts vs flat; CI negative lower |
| 014 | TS momentum long/short | momentum L/S | killed | 0.44 | [0.04, 0.83] | 6.85% | −40.54% | +0.647 | below SPY benchmark; shorting degrades H007 |

---

## Goal trackers

| Target | Current best | Status |
|--------|-------------|--------|
| OOS Sharpe ≥ 1.0 | 0.62 (H007/H008) | in progress |
| Max DD < SPY (−55.92%) | −31.23% (H008) | **met** |
| Strategies tested ≥ 50 | 14 | in progress |
| Strategies kept (ensemble) | 3 | in progress |

---

_See `results/<id>-<name>/stats.md` for full metrics on each hypothesis._
