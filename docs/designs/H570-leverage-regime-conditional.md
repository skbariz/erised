# H570: LeverageRegimeConditional

## Signal
Two-regime leverage using SPY 200d MA as a bull/bear classifier. Bull regime (SPY above
200d MA): cap = 2.5x. Bear regime (SPY below 200d MA): cap = 1.0x (no leverage). Vol-
targeting still applies within both regimes (6.5% target). Same lev-Sharpe SLSQP
objective and MDR warm-start as H559.

## Why should it work?
The 200d MA is one of the oldest and most robust equity market indicators. In bear markets,
leverage amplifies losses and borrow costs are incurred without compensation. By removing
leverage entirely below the 200d MA, the strategy avoids the worst leverage scenarios
(2000-2003, 2007-2009) at the cost of some compounding in minor drawdowns. In bull regimes,
the cap increases from 2.0x to 2.5x — capturing more return during favorable periods. The
net effect should be: better MaxDD (no bear-market leverage) at possibly similar or slightly
lower Sharpe (higher bull-regime leverage compensates for reduced CAGR from no-leverage periods).

## Expected behavior in key regimes
- 2000-2002 tech crash: SPY below 200d MA for extended period → no leverage. Significantly
  better DD. Strategy misses leveraged gains if the portfolio is positioned in TLT/bonds.
- 2008-2009 GFC: similar — bear flag triggers well before the worst of the crash. The H559
  pool includes H473 (L/S) which may be cash-positive in bear markets, but leverage still
  removed.
- 2020 COVID: brief bear regime (Feb-Apr 2020) → no leverage for ~6 weeks. Recovery very
  fast; leverage resumes quickly. Minimal impact.
- 2013-2017 low-vol bull: cap 2.5x fully deployed → higher CAGR than H559's 2.0x cap.
  Max uplift scenario.

## Success criteria
- Sh_lev > 1.79 AND MaxDD_lev < -8% (better MaxDD than H559's -9.73% at comparable Sharpe)
- Ideally: Sh_lev > H556 (1.790) with MaxDD < -7% (combination not yet achieved in Set 8)

## Falsification
- KILL if Sh_lev < H556 (1.790): regime switching adds no Sharpe value and the complexity
  is unjustified.
- Partial KEEP: Sh_lev ≥ 1.79 but MaxDD ≥ -9.73% (same as H559): no MaxDD improvement
  from regime conditioning.

## Results (fill in after running)
- Decision:
- Sh_lev:
- CI:
- CAGR_lev:
- DD_lev:
- Cal_lev:
- Bear-regime fraction (% of time SPY below 200d MA):
- Notes:
