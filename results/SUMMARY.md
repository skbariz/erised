# Strategy Results — Master Leaderboard

Updated after each hypothesis run. All metrics are **walk-forward out-of-sample**
(research period 1993–2022, holdout sealed and untouched). Folds: 5-year train /
2-year test, 12 non-overlapping folds. Costs: 0.5 bps commission + 1.0 bps slippage.

SPY benchmark (same OOS windows): Sharpe 0.51 | CAGR 8.35% | Max DD −55.92%

---

## Leaderboard

| # | Name | Family | Status | OOS Sharpe | 95% CI | CAGR | Max DD | H001 corr (returns) | Notes |
|---|------|--------|--------|-----------|--------|------|--------|----------------------|-------|
| 001 | MA(50,200) crossover | 0.61 | — |

---

## Goal trackers

| Target | Current best | Status |
|--------|-------------|--------|
| OOS Sharpe ≥ 1.0 | 0.62 (H001) | in progress |
| Max DD < SPY (−55.92%) | −33.72% (H001) | **met** |
| Strategies tested ≥ 20 | 7 | in progress |
| Strategies kept (ensemble) | 2 | in progress |

---

_See `results/<id>-<name>/stats.md` for full metrics on each hypothesis._
