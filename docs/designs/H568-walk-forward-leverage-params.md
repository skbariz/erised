# H568: WalkForwardLeverageParams

## Signal
Per-fold leverage parameter selection: in each training fold, sweep 9 combinations of
(target_vol ∈ {5%, 6.5%, 8%}) × (cap ∈ {1.5x, 2.0x, 3.0x}) and select by training-set
lev-Sharpe. Portfolio weights also re-optimized on training data with selected params.
OOS leverage params = training-fold best params. Lookback fixed at 20d.

## Why should it work?
H561 demonstrated that H559's fixed params (6.5%/2.0x/20d) are near the global optimum
for the full OOS sample. But regimes change: the 1993-2002 period (high interest rates,
high realized vol) may need different params than 2010-2022 (low vol, near-zero rates).
Walk-forward param selection adapts to each 5-year training window's vol regime. If the
9-combination grid is small enough, overfitting risk is limited — we're not searching
over thousands of params. The key test: does temporal variation in optimal leverage params
add value over fixed params?

## Expected behavior in key regimes
- 2000-2002: training fold (1993-2000) had moderate vol; selected params likely 6.5-8%/2x.
  OOS 2000-2002 may benefit from slightly lower cap given crash severity.
- 2008-2009: training fold (2002-2007) had low vol; may select high cap (3.0x). If crash
  hits harder than expected, this could hurt. OR: training fold's crash context selects
  conservative params → better protection.
- 2020 COVID: depends on which training fold precedes 2020 test. Short crash may be fine
  regardless of params.
- 2013-2017: consistently low vol; likely selects max leverage (8%/3.0x).

## Success criteria
- Sh_lev > 1.92 (require > 0.04 improvement over H559 = 1.879 to justify overfitting risk)
- CI lower bound > 1.5
- 12/12 folds positive

## Falsification
- Improvement < 0.03 over H559: fixed params are already robust; walk-forward adds no value.
- Improvement positive but < 0.04: not sufficient to justify overfitting risk.
- Negative improvement: per-fold param selection is harmful (training-set overfitting degrades OOS).

## Results (fill in after running)
- Decision:
- Sh_lev:
- CI:
- CAGR_lev:
- DD_lev:
- Cal_lev:
- Params selected per fold:
- Notes:
