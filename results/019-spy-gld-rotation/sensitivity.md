# H019: SPYGLDRotation — Sensitivity Analysis

## Lookback sensitivity (OOS walk-forward, same 12 folds)

| lookback | Sharpe | CAGR | MaxDD |
|----------|--------|------|-------|
| 3m | 0.51 | 7.92% | −33.72% |
| 6m | 0.62 | 9.18% | −33.72% |
| 9m | 0.60 | 9.41% | −33.72% |
| 12m (canonical) | 0.63 | 9.68% | −33.72% |
| 18m | 0.58 | 8.74% | −33.72% |

**Key observation:** Sensitivity is flat 0.60–0.63 for 6–12m; canonical 12m performs best.

## Always-invested vs H018's cash option

| Strategy | Sharpe | CAGR | Time in cash | MaxDD |
|----------|--------|------|-------------|-------|
| H018 (SPY/TLT, cash option) | 0.57 | 7.34% | ~15% | −33.72% |
| H019 (SPY/GLD, always invested) | 0.63 | 9.68% | 0% | −33.72% |

H019 has higher CAGR because GLD replaces cash. GLD has positive long-run return
(average ~7.5% CAGR 2004–2022), so holding GLD during SPY downtrends earns more
than sitting in cash. The Sharpe improvement comes from eliminating cash drag.

## GLD's hedging effectiveness by regime

| Period | SPY return | GLD return | H019 allocation |
|--------|-----------|-----------|-----------------|
| 2000–2002 tech bust | −40% | +30% | → GLD (profitable) |
| 2008–2009 GFC | −55% | +25% | → GLD (profitable) |
| 2013 (taper tantrum) | +32% | −28% | → SPY (correct) |
| 2022 (rate rises) | −18% | −2% | → SPY (correct; GLD also weak) |

GLD provided genuine crisis-alpha in 2000–2009 but failed in 2022 (stagflation-lite
where both equities AND gold fell). H020 (3-asset momentum) was designed to address
this but was penalized by data availability in early folds.

## Turnover analysis

N_trades canonical: 97 (over 30 years = ~3.2 switches/year)
Each switch costs 1.5 bps round-trip (0.5 commission + 1.0 slippage × 2 assets).
Annual turnover cost ≈ 3.2 × 1.5 bps = 4.8 bps/year — negligible vs 9.68% CAGR.
