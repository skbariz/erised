# H218 — IntangibleReturnReversal

## Hypothesis

Construct an "intangible return" = 12-month price return minus YoY NI growth. A stock with high intangible return has price appreciation not supported by earnings growth — it is priced on hope, narrative, or speculative demand. A stock with low intangible return has price that tracks or lags fundamentals — it is "cheap" relative to its earnings trajectory. Long the bottom 20% by intangible return (price grounded in fundamentals); fade the hype-driven stocks.

Conceptually: this is a value/quality hybrid signal without P/B or P/E ratios, grounded in the spread between what the market is pricing and what earnings deliver.

**Mechanics**: intangible_return = price_return(252d, skip 21) - YoY_NI_growth clipped to [-1, 2]. Sector-neutral bottom 20% (ascending=True → low intangible = buy).

## Expected behavior by regime

- **2000-2002 tech crash**: Tech had enormous price returns vs negligible NI growth → very high intangible return → excluded. Strategy avoids the crash.
- **2008-2009 GFC**: Financials had collapsing earnings; intangible return negative or neutral → selected. Banks recover post-2009 quickly.
- **2020 COVID**: Short-term disruption; most NI growth goes to zero, so intangible return tracks price return. Behavior similar to momentum.
- **Low-vol 2013-2017**: Steady compounders with earnings growth track their price → low intangible return → selected.

## Success criteria

- OOS Sharpe > 0.51 with CI lower bound > 0.
- H001 corr near zero (the fundamental anchor should decorrelate from trend-following).
- Max DD < SPY -55.92% (long fundamentally grounded stocks should be more resilient).

## Falsification

If CI lower ≤ 0 or Sharpe < 0.51, the intangible return spread does not predict future returns. If Max DD is substantially worse than SPY, selecting by low intangible return inadvertently concentrates in distressed sectors.

## Results (2026-05-25)

**Decision: KEEP** (marginal — CI lower bound 0.494, barely positive)

| Metric | Value |
|---|---|
| OOS Sharpe | 0.897 |
| 95% CI | [0.494, 1.307] |
| CAGR | 19.11% |
| Max DD | -56.47% |
| H001 corr | -0.001 |

**Per-fold Sharpes**: 0.995, 1.508, 0.816, 1.549, 1.051, 0.541, 0.764, 1.888, 0.775, 1.785, 1.018, 0.980

Negative H001 corr (-0.001) confirms near-zero relationship to equity trend. Max DD -56.47% is marginally worse than SPY (-55.92%) — the strategy holds fundamentally grounded stocks that still decline in market crashes. CI lower bound 0.494 is close to the kill threshold; treat as a weak KEEP. The 2008-2009 regime expectation partially failed: financial stocks with NI collapse had low intangible return and were selected, but then declined severely in the crash.
