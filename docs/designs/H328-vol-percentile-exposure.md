# H328: VolPercentileExposure -- 3-state percentile-rank vol regime with asymmetric scaling

## Hypothesis
H086 applies vol-targeting continuously: even in historically low-vol environments it deflates exposure if the current vol reading (however low) is above 8% annualized. But empirically, when portfolio vol is in its bottom third of its own historical distribution, there is no reason to cut exposure and doing so sacrifices CAGR for no DD benefit. Conversely, simple vol-targeting may not cut exposure aggressively enough when vol enters its top third, because the 8% target implies a linear inverse relationship rather than a floor. This strategy uses the 252-day rolling percentile rank of 21d portfolio vol to define three regimes: below 33rd percentile (scale to 120%), 33rd-67th percentile (scale 1.0), and above 67th percentile (scale = min(0.60, 8%/port_vol) - both a hard floor and a vol-target). The percentile approach is distribution-adaptive: the thresholds adjust automatically across structural vol regime shifts. This design builds on the "vol-of-vol" regime literature (Bollerslev et al., 2009) that shows percentile-based signals outperform fixed-threshold signals out of sample.

**Expected behavior:**
- 2000-2002 tech crash: vol enters top-third percentile early; scale capped at min(0.60, target/vol); significant de-risking; MaxDD better than H086
- 2008-2009 GFC: sustained top-third regime; scale floors at 0.60 with vol-target override; more aggressive than H086 in cutting exposure
- 2020 COVID: rapid shift to top-third; scale drops immediately; 120% bonus in subsequent low-vol period (late 2020 / 2021) accelerates recovery
- 2013-2017 low vol: persistent bottom-third percentile (vol historically low); 120% scale boosts CAGR meaningfully; expected to outperform H086 CAGR over this period

**Success criteria:** OOS Sharpe >= H086 (1.109); MaxDD <= -12%; CI lower > 0.
**Falsification:** Percentile rank adds noise vs fixed thresholds; 120% leverage in low-vol regimes increases CAGR but also increases tail events that dwarf the CAGR gain; or high-regime floor of 0.60 is too lenient compared to H086's vol-target, worsening MaxDD.
