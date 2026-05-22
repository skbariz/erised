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
| 015 | Dual MA L/S (20/100) | trend L/S | killed | 0.08 | [−0.33, 0.50] | −0.28% | −51.00% | +0.213 | near-zero; whipsaw dominates; lowest H001 corr |
| 016 | Volatility-scaled trend | risk mgmt | kept | 0.55 | [0.14, 0.95] | 5.33% | −26.30% | +0.788 | best max DD in program; no ensemble value |
| 017 | Trend L/S + VIX gate | trend / hybrid | killed | 0.43 | [0.03, 0.83] | 5.92% | −39.10% | +0.701 | below SPY benchmark; VIX gate too rare to help |
| 018 | SPY/TLT dual momentum | multi-asset rotation | kept | 0.57 | [0.17, 0.98] | 7.34% | −33.72% | +0.687 | beats benchmark; high H001 corr limits ensemble value |
| 019 | SPY/GLD rotation | multi-asset rotation | kept | 0.63 | [0.22, 1.03] | 9.68% | −33.72% | +0.788 | highest CAGR in rotation batch; too correlated for ensemble |
| 020 | Best-of-3 momentum | multi-asset rotation | killed | 0.47 | [0.06, 0.86] | 6.04% | −31.29% | +0.329 | Sharpe < 0.51; early folds starved of TLT/GLD data |
| 021 | SPY/TLT risk parity | risk parity | **kept ★** | **1.03** | **[0.62, 1.44]** | 7.69% | **−18.41%** | +0.407 | **FIRST OOS Sharpe ≥ 1.0; lowest MaxDD in program** |
| 022 | Defensive equity hedge | multi-asset trend | kept | 0.68 | [0.25, 1.07] | 9.60% | −33.72% | +0.912 | best trend Sharpe; near-zero diversification value |
| 023 | Sell in May | calendar | kept | 0.53 | [0.12, 0.93] | 6.71% | −34.82% | +0.505 | confirmed Bouman & Jacobsen; lower corr than all prior kept strategies |
| 024 | Turn of Month (3d) | calendar | killed | 0.41 | [0.02, 0.82] | 3.72% | −34.41% | +0.363 | Sharpe < 0.51; n_days=5 version (0.54) narrowly beats benchmark |
| 025 | January Barometer | calendar | kept | 0.53 | [0.12, 0.94] | 5.47% | −29.53% | +0.565 | second-best MaxDD (−29.53%); avoids bear years via Jan filter |
| 026 | Quarter-End Drift (5d) | calendar | killed | 0.30 | [−0.12, 0.72] | 1.42% | **−10.13%** | **+0.176** | lowest MaxDD and H001 corr in program; CI negative — standalone fails |
| 027 | Seasonality + MA trend | hybrid calendar | kept | 0.61 | [0.19, 1.03] | 5.70% | −33.72% | +0.712 | matches H001 Sharpe; too correlated for ensemble |
| 028 | VIX spike entry (30, 20d) | VIX contrarian | killed | 0.36 | [−0.07, 0.75] | 4.34% | −44.68% | +0.341 | below benchmark; long during crash continuation |
| 029 | VIX regime filter (<20) | VIX regime | killed | 0.43 | [0.01, 0.84] | 3.39% | −25.53% | +0.562 | below benchmark; CI barely positive; good MaxDD |
| 030 | VIX trend (L/S, MA 20d) | VIX L/S | killed | −0.09 | [−0.50, 0.31] | −3.65% | −83.24% | **−0.284** | negative Sharpe; first negative H001 corr in program |
| 031 | Realized vs implied vol | VIX regime | **kept** | **0.58** | **[0.16, 0.96]** | 8.52% | −51.43% | +0.693 | vol risk premium filter; best CAGR in batch |
| 032 | VIX mean reversion (2σ, 60d) | VIX contrarian | killed | −0.001 | [−0.43, 0.40] | −0.52% | −35.32% | +0.420 | near-zero Sharpe; indistinguishable from zero |

---

## Goal trackers

| Target | Current best | Status |
|--------|-------------|--------|
| OOS Sharpe ≥ 1.0 | **1.03 (H021)** | **MET ★** |
| Max DD < SPY (−55.92%) | **−10.13% (H026, killed)** / **−18.41% (H021, kept)** | **met** |
| Strategies tested ≥ 50 | 32 | in progress |
| Strategies kept (ensemble) | 11 (H001/H007/H008/H016/H018/H019/H021/H022/H023/H025/H027/H031) | in progress — corr still > 0.30; H030 (kill) first negative H001 corr (−0.284) |

---

_See `results/<id>-<name>/stats.md` for full metrics on each hypothesis._
