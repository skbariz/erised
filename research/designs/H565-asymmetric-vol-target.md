# H565: AsymmetricVolTarget

## Signal
Asymmetric vol-targeting leverage: de-risk quickly (5d lookback) when volatility is rising,
but re-lever slowly (20d lookback) when volatility is falling.

Rule: `vol_eff = max(vol_fast_5d, vol_slow_20d)` — always use the more conservative estimate.
Leverage = min(target_vol / vol_eff, max_lev). Same H559 params: 6.5%/2.0x.

lev-Sharpe objective only. MDR warm-start.

## Why should it work?
Standard vol-targeting uses a fixed lookback (20d in H556/H559) which symmetrically adjusts
leverage. The key risk is being slow to de-lever at the start of a crash (vol rises faster
than the lookback window captures). An asymmetric approach that uses a 5d fast estimate when
it is worse than the 20d slow estimate responds to vol spikes within ~1 week rather than ~1
month. The expected benefits:
- During vol spikes (2008-03, 2020-02): faster de-levering reduces left-tail exposure.
- During calm markets (2013-2017): 20d slow estimate is used — same as standard; no change.
- Net effect: lower average leverage (compared to symmetric 20d), lower MaxDD, possibly lower
  CAGR, but better Sharpe due to reduced crash costs.

## Expected behavior in key regimes
- 2000-2002 tech crash: asymmetric approach fires on the initial vol spike in each down leg,
  reducing leverage before the larger drawdown materializes. Should materially reduce fold 0/1 DD.
- 2008-2009 GFC: fastest crash in program history. 5d fast lookback fires within the first week
  of September 2008, cutting leverage significantly before the worst October drawdown.
- 2020 COVID: single fast vol spike in Feb-Mar 2020. Asymmetric approach reacts within 5 days.
  Symmetric 20d would still be averaging in prior low-vol.
- 2013-2017 low-vol: slow 20d estimate dominates; leverage stays near 2.0x; no penalty.

## Success criteria
- Sh_lev > 1.820 AND DD_lev > -8.5% (Pareto-dominates BOTH H559 lev-Sharpe and lev-Composite)
- CI lower bound > 0
- avg_lev lower than H559's 1.23x (confirms faster de-levering in practice)
- All 12 OOS folds positive

## Falsification
If Sh_lev ≤ 1.820 or DD_lev ≤ -8.5%, the asymmetric approach fails to Pareto-dominate.
If avg_lev is similar to H559's 1.23x, the faster de-levering is not triggering in practice
(vol spikes are too short-lived to show a difference over the 30-year OOS window).

## Results (fill in after running)
- Decision:
- Sh_lev:
- CI:
- CAGR_lev:
- DD_lev:
- avg_lev:
- Notes:
