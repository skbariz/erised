# H028: VIXSpikeEntry — Sensitivity Analysis

## Parameter sensitivity (OOS walk-forward, same 12 folds)

### vix_threshold sensitivity (hold_days=20 fixed)

| vix_threshold | Triggers/yr | Sharpe | CI | MaxDD |
|---------------|-------------|--------|----|-------|
| 25 | ~7 | 0.44 | [0.03, 0.84] | −40.12% |
| 30 (canonical) | ~4 | 0.36 | [−0.07, 0.75] | −44.68% |
| 40 | ~1 | 0.16 | [−0.22, 0.52] | −31.40% |

### hold_days sensitivity (vix_threshold=30 fixed)

| hold_days | Sharpe | CI | MaxDD |
|-----------|--------|----|-------|
| 10 | 0.28 | [−0.13, 0.68] | −46.22% |
| 20 (canonical) | 0.36 | [−0.07, 0.75] | −44.68% |
| 30 | 0.39 | [0.00, 0.78] | −43.01% |

**Key observation:** Lower thresholds (25) include more frequent smaller spikes,
improving Sharpe to 0.44 — but still below the 0.51 benchmark. Higher thresholds
(40) are too rare to generate meaningful returns. Longer hold windows improve the
Sharpe marginally (hold_days=30: 0.39) by holding through the initial bounce. None
of the parameter combinations reaches the 0.51 threshold.

## Why the contrarian signal fails

The intuition (VIX spike = equity bottom) is correct on average but too imprecise:

| VIX spike event | VIX peak | SPY behavior during hold_days |
|-----------------|----------|-------------------------------|
| Sep 2001 | 44 | Continued falling after spike |
| Oct 2002 | 40 | Partial bounce, then more decline |
| Oct 2008 | 79 | Continued falling 6 more weeks |
| Aug 2011 | 47 | Partial bounce, then recovery |
| Feb 2018 | 37 | Quick recovery (good timing) |
| Mar 2020 | 66 | Bottom near spike, sharp recovery |

The strategy captures 2018 and 2020 (fast V-shape recoveries) but is long during the
worst multi-week crashes of 2001, 2002, and 2008 where the spike was a midpoint,
not the bottom. The fixed hold_days window cannot distinguish fast vs slow recoveries.

## Comparison to H032 (VIXMeanReversion)

Both H028 and H032 attempt contrarian VIX-based equity entry. H028 uses an absolute
level threshold; H032 normalizes by recent VIX history. Neither works OOS:

| Strategy | Approach | Sharpe | MaxDD |
|----------|----------|--------|-------|
| H028 | Absolute threshold (VIX > 30) | 0.36 | −44.68% |
| H032 | Z-score threshold (z > 2.0) | −0.001 | −35.32% |

H028 actually outperforms H032 despite H032's normalization advantage — the absolute
threshold (30) has more practical relevance (market participants react to round numbers).

## Per-fold performance

| Fold | Period | Sharpe | Dominant event |
|------|--------|--------|----------------|
| 0 | 1998–2000 | 0.62 | Dot-com rise: few spikes, well-timed |
| 1 | 2000–2002 | −0.29 | Tech bust: long during multi-month crash |
| 2 | 2002–2004 | 0.38 | Late bust recovery |
| 3 | 2004–2006 | 0.52 | Low-vol period: few triggers |
| 4 | 2006–2008 | 0.31 | Pre-GFC spike exposure |
| 5 | 2008–2010 | −0.52 | GFC: long during worst drawdown |
| 6 | 2010–2012 | 0.60 | European crisis spikes: good timing |
| 7 | 2012–2014 | 0.12 | Low-vol: insufficient activity |
| 8 | 2014–2016 | 0.43 | China Aug 2015 spike: partial capture |
| 9 | 2016–2018 | 0.11 | Low-vol bull: very few triggers |
| 10 | 2018–2020 | 0.32 | Feb 2018 and COVID: mixed |
| 11 | 2020–2022 | 0.64 | COVID recovery + 2022 spikes |

## Limitations

- **Structural timing risk**: fixed hold window cannot adapt to crash duration
- **H001 correlation +0.341**: meaningfully below median (0.5-0.7 range), but Sharpe is insufficient for standalone or ensemble contribution
- **Cascade crashes**: VIX spike during multi-week selloffs (2008) puts the strategy long during the continuation
