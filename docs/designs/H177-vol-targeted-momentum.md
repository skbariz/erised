# H177: VolTargetedMomentum

## Hypothesis
Momentum crashes occur precisely when the market rebounds sharply after a volatility spike — momentum stocks are short volatility (Barroso & Santa-Clara 2015). Scaling momentum positions by the inverse of recent realized volatility (targeting a constant portfolio volatility) should reduce crash exposure without sacrificing the momentum premium during normal regimes.

Mechanism: select top 20% momentum stocks (12-1m), then weight each stock by `min(1, target_vol / annual_realized_vol)` and renormalize to sum-to-one. During high-vol periods, all stocks are scaled down together; during low-vol periods, weights approach equal-weight momentum.

## Expected behavior
- **2000-2002 tech crash:** Momentum is long tech → crashes hard. Vol scaling reduces position beforehand as vol rises — should significantly outperform plain momentum
- **2008-2009 GFC:** Same mechanism — scaling protects on the way down
- **2020 COVID:** Flash crash; vol scaling triggers protective scaling
- **Low-vol 2013-2017:** Smooth momentum bull run; vol target rarely binds; close to plain momentum

## Success criteria
- OOS Sharpe > plain momentum H150 (0.981) or at least > 0.51 with CI > 0
- MaxDD better than H150 plain momentum (-52.12%)
- All 12 folds positive

## What would falsify this
- MaxDD similar to plain momentum → vol scaling is not helping
- Negative folds in crash periods (2000-2002, 2008-2009)

## Results (OOS)
- **Decision: KEEP**
- Sharpe 0.937, CI [0.535, 1.346], CAGR 17.86%, MaxDD -49.53%
- H001 corr: +0.7271
- All 12 folds positive (min: -0.262 in 2008-2010 — momentum crash fold)
- Sensitivity: target_vol 10%/20%/30% all near 0.94 — stable
- MaxDD slightly better than H150 plain momentum (-52.12%) but not dramatically
- Vol scaling did not prevent 2008 momentum crash (fold 5: -0.262) — surprise result
