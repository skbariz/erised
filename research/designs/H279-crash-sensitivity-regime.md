# H279: CrashSensitivityRegime — Low Coskewness with Regime Tightening

## Hypothesis

Low coskewness stocks (H277) provide diversification value that is most valuable during
stress. In calm regimes, coskewness differences are small and noisy; in stress, they are
large and informative. H279 uses SPY 21d annualized realized vol as a VIX proxy — when vol
exceeds 30% (approximate VIX>25 threshold), tighten the screen from top-quintile to
top-decile on coskewness AND additionally exclude top-decile CVaR names.

Signal:
- Base: coskewness rank ascending, top-quintile (same as H277)
- Stress regime (SPY 21d ann. vol > 0.30): switch to top-decile coskewness only,
  and additionally require CVaR rank ≤ median (bottom-half CVaR excluded)
- Sector-neutral, monthly rebalance

This is the VIX regime filter from the original plan (VIX unavailable for individual stocks).
SPY vol is the cleanest available proxy for market-wide stress.

Expected regime behavior:
- **2000–2002**: Stress regime fires in 2000-2002; tighter screen protects capital.
- **2008–2009**: Stress regime fires hard; tight decile screen is very defensive.
- **2013–2017**: Calm regime dominates; standard quintile sort.
- **2020**: Stress fires in Feb-Mar; tight screen then relaxes in May as vol subsides.

Success criteria: OOS Sharpe ≥ 0.80, CI lower > 0, beats H277 (coskewness without regime).
Falsification: Regime switch hurts Sharpe vs. H277 — the tightening removes good names, not
just bad ones, in stress periods.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.929 |
| 95% CI | [0.522, 1.336] |
| CAGR | 18.27% |
| Max DD | -54.83% |
| H001 corr | -0.002 |
| Decision | KEEP |
