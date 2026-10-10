# H029: VIXRegimeFilter — Sensitivity Analysis

## vix_threshold sensitivity (OOS walk-forward, same 12 folds)

| vix_threshold | Active fraction | Sharpe | CI | MaxDD |
|---------------|----------------|--------|----|-------|
| 15 | ~45% | 0.43 | [0.04, 0.82] | −19.82% |
| 20 (canonical) | ~60% | 0.43 | [0.01, 0.84] | −25.53% |
| 25 | ~72% | 0.32 | [−0.08, 0.72] | −33.14% |

**Key observation:** Lower thresholds are not worse by Sharpe — VIX < 15 (Sharpe 0.43)
matches VIX < 20 while having far lower MaxDD (−19.82% vs −25.53%). The canonical
VIX < 20 uses the well-known "fear gauge" threshold but both fail the 0.51 benchmark.
Higher threshold (VIX < 25) unexpectedly *reduces* Sharpe by capturing too many
elevated-vol periods that don't have strong equity returns.

## Why VIX threshold 25 underperforms 20

Counter-intuitive result: filtering more permissively (VIX < 25 = more active days)
yields *lower* Sharpe than the stricter filter (VIX < 20). Explanation:

- VIX 20-25 zone is "transitional" — moderate fear where markets are choppy, not trending
- VIX < 20 correctly excludes these choppy transition periods
- VIX > 25 correctly excludes crash environments
- The 20-25 zone has lower Sharpe than both the <20 and >25 regimes

## MaxDD comparison

| Strategy | MaxDD | Comment |
|----------|-------|---------|
| H021 RiskParity | −18.41% | Best kept strategy |
| H029 (VIX<15) | −19.82% | Near-equivalent MaxDD (not evaluated as canonical) |
| H026 QuarterEndDrift | −10.13% | Best killed strategy |
| **H029 (VIX<20)** | **−25.53%** | Second-best MaxDD among VIX strategies |
| H025 JanuaryBarometer | −29.53% | Calendar KEEP strategy |
| SPY buy-and-hold | −55.92% | Benchmark |

H029 with VIX<15 achieves a MaxDD of −19.82%, almost matching H021 RiskParity
(−18.41%) — but Sharpe is only 0.43. The risk protection is real; the return is insufficient.

## Regime analysis: what does VIX<20 capture?

| Period | VIX regime | H029 active? | SPY performance |
|--------|-----------|-------------|-----------------|
| 1993–1997 | Mostly VIX<20 | Yes | Strong bull |
| 1998–2002 | VIX spiked 30-40 | No (crash periods) | Tech bust losses |
| 2003–2007 | VIX mostly <20 | Yes | Recovery rally |
| 2008–2009 | VIX 40-80 | No | GFC losses |
| 2010–2019 | Mostly VIX<20 | Yes | Bull market |
| 2020 | VIX 20-80 | Mixed | COVID crash + recovery |
| 2022 | VIX 20-35 | Mostly no | Bear market losses |

The filter correctly avoids 2008-2009 and 2022. The problem: Sharpe of active
periods (VIX<20 regime) is only ~0.65, not high enough to offset the missed returns
from always being flat in elevated-VIX periods.

## Per-fold analysis

| Fold | Period | Sharpe | Comment |
|------|--------|--------|---------|
| 0 | 1998–2000 | 0.82 | VIX mostly low; missed 1998 spike |
| 1 | 2000–2002 | 0.28 | Tech bust: frequently flat (VIX>20) |
| 2 | 2002–2004 | 0.35 | Recovery: mixed signals |
| 3 | 2004–2006 | 0.54 | Low-vol era: mostly active |
| 4 | 2006–2008 | 0.60 | Pre-GFC calm |
| 5 | 2008–2010 | −0.04 | GFC: flat most of time but some false entries |
| 6 | 2010–2012 | 0.56 | Post-GFC recovery |
| 7 | 2012–2014 | 0.41 | Bull market |
| 8 | 2014–2016 | 0.53 | Consistent |
| 9 | 2016–2018 | 0.52 | Low-vol bull |
| 10 | 2018–2020 | 0.26 | 2018 correction + COVID mixed |
| 11 | 2020–2022 | 0.47 | 2022: correctly flat but 2021 gained less |

## Kill analysis and ensemble note

Kill confirmed: Sharpe 0.43 < 0.51 threshold. However:
- MaxDD −25.53% is significantly better than SPY (−55.92%)
- CI lower bound +0.01 is barely positive — one of the less-certain kills
- H001 correlation +0.562: moderate, same level as H023 SellInMay (kept)

The strategy fails the Sharpe threshold by a small margin. A modified version with
a 5-day smoothed VIX (to reduce whipsaw near the 20 threshold) might bridge the gap.
The 314 trades represent frequent threshold crossings — smoothing the signal would
reduce turnover cost and possibly improve net Sharpe.
