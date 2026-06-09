# H567: LevSortinoDownsideVol

## Signal
Replace standard total-vol targeting with downside-vol targeting (semi-deviation only).
Leverage = target_vol / downside_std, where downside_std uses only negative return days
(annualized with sqrt(2) adjustment). SLSQP objective maximizes Sortino ratio of the
downside-vol-targeted portfolio. MDR warm-start. H559 leverage params: 6.5%/2.0x/20d.

## Why should it work?
Standard vol-targeting de-levers during volatile upswings (e.g. recovery rallies after
crashes) — penalizing returns when risk is actually low. Downside vol targeting only
de-levers when negative tail risk is elevated. The sqrt(2) adjustment ensures that for
normally distributed returns, downside vol ≈ total vol (no systematic bias). Combined
with a lev-Sortino objective, the optimizer rewards return per unit of downside risk,
which should find weights that pair well with downside-selective de-levering. The
mechanism may produce higher CAGR in trending/recovering markets (less unnecessary de-
levering) while retaining crash protection (de-levers in actual bear markets).

## Expected behavior in key regimes
- 2000-2002 tech crash: downside vol spikes → leverage cuts cleanly (same as total-vol
  targeting). Crash protection intact.
- 2008-2009 GFC: same as above — high downside vol drives de-levering. Expect similar
  or better DD than H559.
- 2020 COVID: short crash + fast recovery. Standard vol stays high during recovery (total
  vol elevated); downside vol drops faster → more leverage during recovery → higher CAGR
  vs H559.
- 2013-2017 low-vol: upside skew periods; both vol measures low → leverage at cap. No
  material difference from H559.

## Success criteria
- Sh_lev > 1.879 (H559 champion) OR Cal_lev > 1.802 (H558 CAGR champion)
- CI lower bound > 1.4 (stricter — downside-vol overfitting risk)
- 12/12 folds positive (fold consistency)

## Falsification
- If Sh_lev < 1.790 (H556): downside-vol targeting adds no Sharpe value vs standard
  vol-targeting.
- If Sh_lev ≥ 1.790 but not > H559 (1.879): marginal improvement insufficient to justify
  complexity.

## Results (fill in after running)
- Decision:
- Sh_lev:
- CI:
- CAGR_lev:
- DD_lev:
- Cal_lev:
- avg_lev (downside vs standard comparison):
- Notes:
