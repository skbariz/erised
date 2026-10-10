# H559: LeveragedSharpeMDR

## Signal
Co-adapt portfolio weights to vol-targeting leverage (H556: 6.5%/2.0x/20d) by directly
optimizing the Sharpe of the deployed (leveraged) return stream, rather than applying leverage
post-hoc to a base-Sharpe-optimized portfolio (H556's approach).

## Hypothesis
H558 showed leveraged-Calmar objective beats pure-Calmar objective for Cal_lev (1.802 vs 1.429).
By analogy, leveraged-Sharpe objective should beat H556's approach (which used H551's base-Sharpe
weights + post-hoc leverage) for Sh_lev. H551/H556 baseline Sh_lev = 1.741/1.790.

## Objectives tested
1. Leveraged Sharpe: maximize Sharpe of vol-targeted return stream
2. Leveraged Sortino: maximize Sortino of vol-targeted return stream
3. Leveraged composite: equal weight Sh_lev + Cal_lev + So_lev (all on leveraged returns)

## Expected behavior
- Leveraged-Sharpe optimizer will push toward smooth, consistent-return strategies
  (H473 L/S PCA, H157 vol-scaled momentum) since vol-targeting pairs well with low-vol streams
- Should reduce H001 corr vs H556 (similar to H558's +0.519 vs H556's +0.612)
- In high-vol regimes (2008-2009 GFC, 2020 COVID): vol-targeting de-levers; optimizer
  pre-selects strategies that hold up when deleveraged
- In low-vol regimes (2013-2017): optimizer will allow higher leverage → CAGR benefit

## Success criteria
- Sh_lev > 1.790 (H556 baseline) — would confirm "optimize deployed metric" generalizes
- CI lower bound > 0
- All 12 folds positive

## Falsification
If Sh_lev ≤ 1.790, the benefit of co-adaptation is specific to Calmar objectives and
vol-targeting already captures most of the Sharpe benefit from portfolio composition.
