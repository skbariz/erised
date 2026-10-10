# H032: VIXMeanReversion — Sensitivity Analysis

## Parameter sensitivity (OOS walk-forward, same 12 folds)

### n_sigma sensitivity (lookback=60 fixed)

| n_sigma | Active days/yr | Sharpe | CI | MaxDD |
|---------|---------------|--------|----|-------|
| 1.5 | ~35 | 0.12 | [−0.28, 0.52] | −24.83% |
| 2.0 (canonical) | ~18 | −0.001 | [−0.43, 0.40] | −35.32% |
| 2.5 | ~8 | 0.03 | [−0.38, 0.43] | −18.61% |

### lookback sensitivity (n_sigma=2.0 fixed)

| lookback | Sharpe |
|----------|--------|
| 20d | 0.08 |
| 60d (canonical) | −0.001 |
| 120d | −0.04 |

**Key observation:** Lower n_sigma (1.5) slightly improves Sharpe to 0.12 — more frequent
signals but still far below the 0.51 threshold. Extremely high threshold (2.5) is too
rare to generate returns. The canonical n_sigma=2.0 is the worst choice (nearly zero
Sharpe) — the sweet spot may be in the 1.5-1.8 range, but none produces acceptable
OOS performance.

**Surprising finding:** n_sigma=2.5 has the lowest MaxDD (−18.61%) — active only ~8
days/year means almost zero drawdown exposure. But with such rare activation, the CAGR
is effectively zero and the strategy has no standalone value.

## Why the contrarian VIX signal fails where H028 (absolute threshold) fails too

Both H028 and H032 exploit the same intuition: extreme VIX → equity bottom. H032
normalizes by the current vol regime; H028 uses an absolute threshold. Both fail:

| Strategy | Normalization | Sharpe | MaxDD |
|----------|--------------|--------|-------|
| H028 (absolute: VIX > 30) | None | 0.36 | −44.68% |
| H032 (z-score: z > 2.0) | 60d rolling | −0.001 | −35.32% |

The z-score normalization in H032 activates during a VIX spike of 12 in a low-vol
regime (e.g., VIX normally at 11, spikes to 12 = +1.0σ above 60d mean). This creates
false contrarian signals in stable low-vol markets where the "spike" is trivial.

Conversely, during 2008 when VIX rose from 25 to 80, the 60d mean was tracking up with
it. The z-score was elevated but less extreme than the absolute level suggested — the
normalization attenuated the strongest contrarian signals.

## Market microstructure: VIX is mean-reverting but timing is hard

VIX mean-reversion is well-documented (half-life ~20 days), but:

1. **Mean-reversion from what level?** After a spike to 50, the mean is not at the
   pre-spike level (e.g., 15) — it reverts to a post-crisis elevated mean (~25-30)
   over many months.

2. **The reversion often takes longer than the hold window**: H032 has no explicit
   hold window — it's only active when z > threshold. Once VIX normalizes (z falls
   below threshold), the position exits, potentially before the equity bounce.

3. **False positives in low-vol regimes**: Small VIX moves (e.g., 12 → 15 in a
   quiet market) can trigger the 2σ threshold, leading to equity entries that are
   not genuine fear-based opportunities.

## Comparison to H028

H028 (absolute threshold) achieves Sharpe 0.36 by focusing on genuinely extreme fear
events (VIX > 30 is a known market stress indicator). H032's z-score normalization
inadvertently includes smaller relative spikes in normally-calm periods, diluting the
signal quality. The absolute threshold has more practical anchoring.

## Per-fold analysis

| Fold | Period | Sharpe | Comment |
|------|--------|--------|---------|
| 0 | 1998–2000 | 0.27 | LTCM spike: good timing |
| 1 | 2000–2002 | −0.51 | Tech bust: long during crash |
| 2 | 2002–2004 | 0.24 | Late bust: some recovery |
| 3 | 2004–2006 | 0.13 | Low-vol: few triggers |
| 4 | 2006–2008 | 0.14 | Pre-GFC |
| 5 | 2008–2010 | −0.22 | GFC: active during worst phases |
| 6 | 2010–2012 | 0.08 | European debt: marginal |
| 7 | 2012–2014 | −0.07 | Low-vol bull: few signals |
| 8 | 2014–2016 | 0.13 | China Aug 2015: partial |
| 9 | 2016–2018 | −0.33 | Low-vol era: false positive spikes |
| 10 | 2018–2020 | 0.31 | Feb 2018 + COVID partial |
| 11 | 2020–2022 | 0.20 | Mixed |

**Five negative folds**: the strategy fails in exactly the regimes where H028 also fails
(2000-2002 tech bust, 2008 GFC, 2016-2018 low-vol bull).

## Limitations

- Killed with near-zero Sharpe (−0.001); essentially a zero-return strategy OOS
- Z-score normalization creates false signals in stable-VIX environments
- Structural overlap with H028 in concept; H028 has marginally better OOS performance
- Neither contrarian VIX strategy (H028, H032) demonstrates sufficient timing precision
  for profitability at daily frequency
