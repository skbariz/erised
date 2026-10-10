# H340: GrandCrisisProtocol -- Union of four crisis detectors, tiered response

## Hypothesis
Individual crisis detectors (H097 correlation, H079 drawdown+VIX, H306 macro stress, H301 VRP) each capture different dimensions of market crisis. No single detector is universally correct — the 2000-2002 crisis was primarily detected by drawdown and momentum; the 2008-2009 GFC by correlation contagion and credit stress; the 2020 COVID crash by VRP and VIX. This strategy aggregates all four into a tiered response: 2+ signals → maximum defense (20% SPY / 80% TLT); 1 signal → moderate defense (40% SPY / 60% TLT); 0 signals → H021 base. The voting approach prevents single-signal false positives from triggering maximum defense while ensuring that when multiple independent detectors fire simultaneously, the system responds decisively. Pre-TLT era: defaults to SPY only.

**Expected behavior:**
- 2000-2002: Drawdown+VIX (S2) fires progressively; macro stress (S3) escalates; 2+ signals → 20/80 TLT; strong protection
- 2008-2009: All four signals fire simultaneously at peak crisis → 20/80 throughout 2008-2009 depth → maximum protection
- 2013-2017 low vol: All four signals at zero throughout → pure H021; no friction cost
- 2020 COVID: All four fire in Feb-March 2020; correlation (S1) + VIX/DD (S2) + stress (S3) + severe VRP (S4) → 20/80 → sharp but brief protection; recovery when signals reset

**Success criteria:** OOS Sharpe > 0.90; MaxDD < -15%; CI lower > 0; Sharpe should match or beat H079/H097 as crisis protection improves.
**Falsification:** The four-signal voting creates too many simultaneous false alarms in non-crisis periods (signals correlated with each other), or the 20/80 floor during crises misses recovery and costs returns when one signal resets before others.

## Results
*[To be filled after backtest]*
Decision:
