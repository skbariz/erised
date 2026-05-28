# H326: DualFrequencyVolTarget -- Conservative vol-target using max(21d, 63d) portfolio vol

## Hypothesis
H086 (LowVolTargetPortfolio) applies a 21-day realized vol estimate to scale H021 weights down when vol spikes. A known failure mode is that short-window vol estimates mean-revert quickly, so by the time the 21d signal fires the worst of the vol spike may have passed, but the position has already been cut. Conversely, a 63-day window reacts slowly and may miss fast-moving vol events (e.g., COVID crash in February-March 2020). This strategy uses max(21d_port_vol, 63d_port_vol) as the denominator in the scaling formula: scale = min(1, 8% / max(21d_vol, 63d_vol)). During an accelerating vol regime, the 21d vol rises above the 63d vol and drives the scale down faster; during a decelerating regime the 63d vol (which has "remembered" the spike) keeps the scale lower for longer. This asymmetric conservatism is motivated by Engle's (2002) finding that volatility is long-memory and post-spike persistence is underestimated by short windows.

**Expected behavior:**
- 2000-2002 tech crash: 21d vol spikes well above 63d -> scale contracts rapidly; protection matches or beats H086
- 2008-2009 GFC: both vols high; the 63d vol remains elevated into recovery, delaying re-entry compared to H086; MaxDD likely similar but recovery slightly slower
- 2020 COVID: 21d vol spikes far above 63d; fires quickly; scale drops hard; 63d then holds it down through April-May 2020; more conservative than H086 through that period
- 2013-2017 low vol: 21d < 63d -> max() dominated by 63d; scale slightly more conservative than pure 21d H086; small CAGR penalty but improved tail protection

**Success criteria:** OOS Sharpe >= H086 (1.109); MaxDD <= -12%; CI lower > 0.
**Falsification:** Dual-window adds no value over H086; Sharpe and MaxDD are within noise of H086 across all three vol_target levels tested in sensitivity, or MaxDD worsens relative to H086 despite conservative sizing.
