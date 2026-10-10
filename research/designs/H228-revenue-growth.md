# H228 — RevenueGrowth

## Hypothesis

YoY top-line revenue growth as a cross-sectional predictor. Revenue growth is a pure measure of business expansion — unlike earnings, it is not affected by cost management, capitalization decisions, or accruals. The hypothesis: companies with stronger revenue growth are gaining market share, entering new markets, or benefiting from secular tailwinds — and the market underreacts to sustained revenue momentum (similar to PEAD but for revenue rather than earnings).

**Mechanics**: rev_growth = (rev - rev.shift(252)) / rev.shift(252), clipped [-1, 3]. Sector-neutral top 20%. Monthly rebalance.

## Expected behavior by regime

- **2000-2002 tech crash**: Tech had strong revenue growth (often unprofitable) — this signal would select into tech pre-crash. Risk factor.
- **2008-2009 GFC**: Revenue collapsed industry-wide; post-2009 recovery selects companies with fastest revenue recovery.
- **2020 COVID**: E-commerce, cloud, healthcare: strongest revenue growth during and after COVID.
- **Low-vol 2013-2017**: Consumer discretionary and tech saw steady revenue growth; this regime favors the signal.

## Success criteria

- OOS Sharpe > 0.51 with CI lower bound > 0.
- H001 corr near zero (no price component).
- Consistent folds 6-11 (signal should be robust across market regimes once EDGAR data is available).

## Falsification

If Sharpe < 0.51, revenue growth is already priced in — high-growth stocks are expensive and their returns reflect the premium paid. The tech crash period (if folds 0-5 were covered) might have killed this signal pre-2009.

## Results (2026-05-25)

**Decision: KEEP**

| Metric | Value |
|---|---|
| OOS Sharpe | 0.760 |
| 95% CI | [0.357, 1.170] |
| CAGR | 8.93% |
| Max DD | -36.62% |
| H001 corr | -0.023 |

**Per-fold Sharpes**: 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.484, 2.138, 0.838, 1.592, 1.006, 1.142

Folds 0-5 zero (EDGAR gap). Folds 6-11 average 1.200. H001 corr -0.023, strongest (most negative) in Batch 6. Max DD -36.62% substantially better than SPY. The 252d window is the sweet spot: shorter (126d) is noisier, longer (504d) too stale. Revenue growth is the most decorrelated quality factor from equity trend-following discovered so far.
