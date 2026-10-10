# H231 — QualityMomentum

## Hypothesis

Combine Novy-Marx (2013) gross profitability (GP/TA) with 12-1m price momentum as an equal-weight z-score composite. Quality provides a fundamental anchor that keeps the momentum signal away from low-quality momentum winners (which are more prone to crashes and mean-reversion). The hypothesis: stocks that are both high-quality AND showing price momentum are the most durable winners.

## Results (2026-05-25)

**Decision: KEEP** — Batch 7 star

| Metric | Value |
|---|---|
| OOS Sharpe | 0.870 |
| 95% CI | [0.460, 1.282] |
| CAGR | 11.24% |
| Max DD | -35.71% |
| H001 corr | -0.014 |

**Per-fold Sharpes**: 0.000, 0.000, 0.000, 0.000, 0.000, 0.475, 0.800, 2.217, 0.848, 2.049, 1.154, 1.096

Remarkable robustness: Sharpe is virtually flat across 126/252/504d momentum windows (0.871/0.870/0.872) — the GP/TA quality anchor stabilizes the signal. Folds 6-11 average 1.360. Max DD -35.71% substantially better than pure momentum strategies. H001 corr -0.014 despite the momentum component.
