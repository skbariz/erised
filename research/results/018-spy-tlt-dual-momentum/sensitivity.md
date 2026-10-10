# H018: SPYTLTDualMomentum — Sensitivity Analysis

## Lookback sensitivity (OOS walk-forward, same 12 folds)

| lookback | skip | Sharpe | CAGR | MaxDD | Note |
|----------|------|--------|------|-------|------|
| 3m | 1m | 0.44 | 6.17% | −33.72% | Too reactive; whipsaw cost |
| 6m | 1m | 0.68 | 9.03% | −33.72% | Best Sharpe; shorter momentum cycle |
| 9m | 1m | 0.53 | 7.46% | −33.72% | Mid-range |
| 12m | 1m | 0.57 | 7.34% | −33.72% | Canonical (Antonacci convention) |
| 18m | 1m | 0.49 | 6.21% | −33.72% | Too slow; misses turning points |

**Key observation:** The 6m lookback outperforms the canonical 12m (0.68 vs 0.57).
The Antonacci convention (12m) is used as canonical to match academic replication.

## Skip-month sensitivity (lookback fixed at 12m)

| skip | Sharpe |
|------|--------|
| 0 (no skip) | 0.51 |
| 1m (canonical) | 0.57 |
| 2m | 0.52 |

The 1-month skip improves Sharpe by avoiding short-term reversal at lag 1.

## Signal stability (bootstrap distribution)

- 95% CI on canonical: [0.17, 0.98]
- Width: 0.81 — moderate uncertainty
- P(Sharpe > 0): ~97.2%
- P(Sharpe > 0.51): ~61.8%

## Regime analysis (per-fold performance)

| Regime | Folds | Avg Fold Sharpe |
|--------|-------|-----------------|
| Tech crash / early 2000s | 2–3 | 1.09 |
| Financial crisis 2008–2009 | 5–6 | 0.47 |
| Post-GFC recovery | 7–8 | 0.90 |
| Low-vol bull (2013–2017) | 9–10 | 1.17 |
| Rate-rise / late-cycle | 11 | 0.50 |

TLT allocation during 2008–2009 GFC provided meaningful protection but the
strategy's MaxDD remained −33.72% — SPY allocation going into the crash before
signals updated caused drawdown comparable to buy-and-hold.

## H001 correlation decomposition

Overall H001 correlation: +0.687

By period:
- Rising markets (SPY allocated): effectively = H001 returns → high correlation
- Falling markets (TLT allocated): uncorrelated or negative → pulls corr down
- Cash periods: zero return → zero correlation contribution

The high overall correlation is structurally inevitable when SPY is the default
allocation and SPY has positive long-run drift. Correlation would fall below 0.30
only if TLT allocation were the majority of the time — which would require a
persistent equity bear market.
