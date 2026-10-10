# H013 — MA(50,200) Long/Short: Stats

**Status:** killed
**Family:** trend / long-short
**Instrument:** SPY
**Run date:** 2026-05-21

## Parameters

| Parameter | Value |
|-----------|-------|
| Fast MA | 50 days |
| Slow MA | 200 days |
| Direction | long/short (±1) |
| Sizing | binary (−1 / +1) |

## Walk-forward configuration

| Setting | Value |
|---------|-------|
| Research period | 1993-01-29 to 2022-12-30 |
| Train window | 5 years |
| Test window | 2 years |
| Folds | 12 |
| Commission | 0.5 bps |
| Slippage | 1.0 bps |

## OOS metrics (combined across 12 folds)

| Metric | H013 | SPY benchmark | Delta |
|--------|------|---------------|-------|
| Sharpe | 0.36 | 0.51 | −0.15 |
| 95% CI on Sharpe | [−0.04, 0.77] | — | — |
| CAGR | 5.32% | 8.35% | −3.03 pp |
| Max drawdown | −40.16% | −55.92% | −15.76 pp |
| Trades | 25 | — | — |

## Sensitivity (fast/slow pair)

| Params | Sharpe |
|--------|--------|
| MA(20,100) | 0.08 |
| MA(50,200) canonical | 0.36 |
| MA(50,150) | 0.34 |

## Per-fold OOS Sharpe

`['0.05', '0.53', '0.67', '0.57', '0.83', '0.63', '-0.48', '1.51', '-0.06', '1.13', '0.24', '-0.13']`

## Correlation with H001: +0.707

## Decision: killed

The short side destroys what H001 earns. H001 (long-only, Sharpe 0.61) outperforms H013 (long/short, Sharpe 0.36) by 25 bps of Sharpe. The 95% CI lower bound is negative (−0.04), indicating the strategy cannot be distinguished from noise across the test period.

The problem: when the MA signal is bearish (fast < slow), adding short exposure rather than going flat means active losses during whipsaw periods — particularly the 2000–2002 and 2008 drawdowns where the signal repeatedly crosses while prices chop. Going flat (H001's approach) is better than going short on a slow-reacting signal. The death cross is effective at avoiding crashes but not at timing them precisely enough for short profits.

Correlation 0.707 with H001 confirms it's measuring the same underlying signal with worse execution.
