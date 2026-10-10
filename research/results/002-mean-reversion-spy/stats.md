# H002 — Mean Reversion after 2σ Down Moves: Stats

**Status:** killed
**Family:** mean reversion
**Instrument:** SPY
**Run date:** 2026-05-21

## Parameters (canonical)

| Parameter | Value |
|-----------|-------|
| Lookback | 20 days |
| Entry threshold (z-score) | −2.0σ |
| Hold days | 5 days |
| Direction | long-only |
| Sizing | binary (0 / +1) |

## Walk-forward configuration

| Setting | Value |
|---------|-------|
| Research period | 1993-01-29 to 2022-12-30 |
| Train window | 5 years |
| Test window | 2 years |
| Folds | 12 (identical boundaries to H001) |
| OOS periods | 6,036 bars |
| Commission | 0.5 bps |
| Slippage | 1.0 bps |

## OOS metrics — canonical (lookback=20, threshold=2.0, hold_days=5)

| Metric | H002 | SPY benchmark | Delta |
|--------|------|---------------|-------|
| Sharpe | 0.24 | 0.51 | −0.27 |
| 95% CI on Sharpe | [−0.17, 0.65] | — | — |
| CAGR | 1.85% | 8.35% | −6.50 pp |
| Ann vol | 9.67% | 19.66% | −10.0 pp |
| Max drawdown | −28.04% | −55.92% | −27.88 pp |
| Calmar | 0.07 | 0.15 | −0.08 |
| Sortino | 0.13 | — | — |
| Hit rate | 47.67% | 54.54% | −6.87 pp |
| Total trades | 370 | — | |
| Turnover / day | 6.13% | — | |

## Ensemble / correlation check

| Measure | Value | Threshold | Pass? |
|---------|-------|-----------|-------|
| Returns corr with H001 | +0.546 | < 0.30 | NO |
| Position corr with H001 | +0.090 | — | — |

## Parameter grid stability (27 combinations)

Grid: lookback ∈ {10,20,30} × threshold ∈ {1.5,2.0,2.5} × hold_days ∈ {3,5,7}

| Metric | Value |
|--------|-------|
| Median OOS Sharpe | 0.24 |
| IQR | [0.14, 0.34] |
| Count Sharpe > 0.5 | 0 / 27 |
| Min / Max | 0.06 / 0.44 |

## 1-D sensitivity

**Lookback** (threshold=2.0, hold_days=5):

| lookback | Sharpe |
|----------|--------|
| 10 | 0.30 |
| **20** | **0.24** |
| 30 | 0.13 |

**Threshold** (lookback=20, hold_days=5):

| threshold | Sharpe |
|-----------|--------|
| 1.5 | 0.36 |
| **2.0** | **0.24** |
| 2.5 | 0.33 |

**Hold days** (lookback=20, threshold=2.0):

| hold_days | Sharpe |
|-----------|--------|
| 3 | 0.06 |
| **5** | **0.24** |
| 7 | 0.24 |

## Per-fold OOS Sharpe

| Fold | Test window | H002 | SPY |
|------|-------------|------|-----|
| 0 | 1998-01-29 to 2000-01-29 | +0.69 | +0.94 |
| 1 | 2000-01-29 to 2002-01-28 | −0.69 | −0.31 |
| 2 | 2002-01-28 to 2004-01-28 | +1.28 | +0.20 |
| 3 | 2004-01-28 to 2006-01-27 | −0.05 | +0.77 |
| 4 | 2006-01-27 to 2008-01-27 | +0.17 | +0.33 |
| 5 | 2008-01-27 to 2010-01-26 | +0.35 | −0.06 |
| 6 | 2010-01-26 to 2012-01-26 | +0.12 | +0.67 |
| 7 | 2012-01-26 to 2014-01-25 | +1.61 | +1.51 |
| 8 | 2014-01-25 to 2016-01-25 | +0.18 | +0.46 |
| 9 | 2016-01-25 to 2018-01-24 | +1.29 | +2.40 |
| 10 | 2018-01-24 to 2020-01-24 | −0.22 | +0.73 |
| 11 | 2020-01-24 to 2022-01-23 | −0.03 | +0.75 |

## Kill reason

1. Weak signal — Sharpe 0.24, CI [−0.17, 0.65] includes zero; 0/27 param
   combinations reach Sharpe 0.5.
2. High returns correlation with H001 (+0.55) — fails ensemble threshold.
