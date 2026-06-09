# H562: High-Cap Leveraged Sharpe (target_vol=5%, cap=3.0×)

## Signal
Lev-Sharpe SLSQP on 55-strategy pool with aggressive leverage params (target_vol=5%, cap=3.0×).
Same pool and warm-start as H559.

## Why should it work?
H559's avg realized leverage was only ~1.23x (far below the 2.0x cap). At a tighter vol target
(5% vs 6.5%) and higher cap (3.0x vs 2.0x), avg leverage should reach ~1.6-1.8x in low-vol
periods, substantially increasing CAGR. The cap increase to 3.0x allows exploitation of the
low-vol regimes H559 left on the table; the lower target vol ensures the optimizer co-adapts
to a more aggressive deployment profile.

## Expected regime behavior
- 2000-2002 tech crash: high realized vol → leverage ~1.0x, similar to H559 (limited damage)
- 2008-2009 GFC: same — vol spikes will clip leverage near 1.0x quickly
- 2020 COVID: sharp vol spike then rapid collapse → leverage ramps fast in recovery (benefit)
- 2013-2017 low vol: leverage reaches 3.0x cap more frequently → major CAGR boost here

## Success criteria
Sh_lev > 1.879 (H559) OR CAGR > 15% at Sh_lev > 1.7.

## Falsification
Higher cap adds leverage without improving Sharpe — more risk than return. If MaxDD > −20%,
the additional leverage is not being compensated by proportional Sharpe improvement.

## Reference
H559: Sh_lev=1.879, CAGR=13.08%, MaxDD=−9.73%, at 6.5%/2.0x/20d.
