# H564: MaxDDConstrainedLevSharpe

## Signal
Lev-Sharpe SLSQP with an explicit MaxDD penalty that targets DD_lev ≤ 7%.
The objective is `-(Sh_lev - λ * max(0, -DD_lev - 0.07))` where λ is swept over
[1.0, 5.0, 10.0, 20.0]. MDR warm-start is used for all four runs.

Best candidate selected by: max Sh_lev subject to DD_lev > -9.0% (strictly better DD
than H559's lev-Sharpe result of -9.73%).

## Why should it work?
H559's lev-Sharpe optimizer achieved Sh_lev=1.879 but DD_lev=-9.73%, which exceeds the
institutional -9.0% target. The lev-Composite from H559 achieved better DD (-7.25%) but
lower Sh_lev (1.762). A penalized Sharpe objective should find a Pareto point between the
two: the optimizer retains the high-Sharpe portfolio structure while being pushed away from
weight combinations that produce large drawdowns. The penalty is applied on the leveraged
return stream (the actually-deployed metric), so the optimizer co-adapts weights to both the
leverage mechanism and the DD constraint simultaneously.

## Expected behavior in key regimes
- 2000-2002 tech crash: penalty pushes optimizer toward H043/H078 (calendar/seasonal), which
  hold TLT during crash months and protect DD. H473 L/S should maintain stable weight.
- 2008-2009 GFC: vol-targeting deleverages strongly; H021 risk parity recovers quickly post-crash.
  A moderate penalty (λ=5-10) likely finds full coverage without over-constraining Sharpe.
- 2020 COVID: sharp but short drawdown; DD penalty fires once then resolves. Low λ handles this.
- 2013-2017 low-vol: leverage rides up; calendar strategies add seasonal alpha. DD is mild.

## Success criteria
- Sh_lev > 1.762 (H559 lev-Composite reference)
- DD_lev > -9.0% (strictly better than H559 lev-Sharpe's -9.73%)
- CI lower bound > 0
- All 12 OOS folds positive

## Falsification
If no penalty level achieves DD_lev > -9.0% while maintaining Sh_lev > 1.762, the lev-Sharpe
objective cannot be meaningfully constrained on DD without collapsing to lev-Composite performance.
This would suggest a true Pareto frontier discontinuity between Sharpe and MaxDD at this pool size.

## Results (fill in after running)
- Decision:
- Best penalty: λ=
- Sh_lev:
- CI:
- CAGR_lev:
- DD_lev:
- Notes:
