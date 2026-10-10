# H014 — TS Momentum Long/Short: Stats

**Status:** killed
**Family:** momentum / long-short
**Instrument:** SPY
**Run date:** 2026-05-21

## Parameters

| Parameter | Value |
|-----------|-------|
| Long lookback | 12 months (252 days) |
| Skip months | 1 (21 days) |
| Direction | long/short (±1) |

## OOS metrics (combined across 12 folds)

| Metric | H014 | SPY benchmark | Delta |
|--------|------|---------------|-------|
| Sharpe | 0.44 | 0.51 | −0.07 |
| 95% CI on Sharpe | [0.04, 0.83] | — | — |
| CAGR | 6.85% | 8.35% | −1.50 pp |
| Max drawdown | −40.54% | −55.92% | −15.38 pp |
| Trades | 97 | — | — |

## Sensitivity (long_months)

| long_months | Sharpe |
|-------------|--------|
| 6 | 0.22 |
| 9 | 0.41 |
| **12** | **0.44** |

## Per-fold OOS Sharpe

`['0.93', '0.37', '0.48', '0.77', '0.33', '0.12', '0.28', '1.43', '0.35', '1.20', '-0.01', '0.33']`

## Correlation with H001: +0.647

## Decision: killed

Sharpe 0.44 is below the SPY buy-and-hold benchmark (0.51), failing the primary keep criterion. Extending H007's momentum signal to go short when momentum is negative — rather than flat — degrades performance: H007 (long-only) earns 0.62 vs H014 (long/short) at 0.44.

The pattern matches H013's finding: the long-only signal is well-calibrated; adding short exposure introduces active losses during momentum-reversal periods that exceed the gains from correctly-timed shorts. The 12-month momentum signal is not precise enough at the turns to profit from shorting.

CI lower bound 0.04 is technically positive but narrow — this is consistent with a marginal positive signal obscured by execution noise from the short side. Correlation 0.647 with H001 confirms shared underlying trend exposure.
