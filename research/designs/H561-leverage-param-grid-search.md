# H561: Leverage Parameter Grid Search

## Signal
Grid search over leverage parameters (target_vol, cap, lookback) for H559's lev-Sharpe SLSQP.

## Why should it work?
H559 uses fixed params (6.5%/2.0x/20d) inherited from H556 which inherited from H086. These
were validated on unleveraged portfolio behavior, not on lev-Sharpe optimization. A systematic
grid search may find params that interact better with the 55-strategy pool.

## Expected behavior
- Shorter lookback (10d): faster vol reaction, may reduce MaxDD in crash events
- Higher cap (3.0x): more CAGR in low-vol periods, potentially worse MaxDD
- Higher target_vol (8%): more leverage overall, higher CAGR, higher MaxDD

## Success criteria
Any combination improves on H559 (Sh_lev=1.879) by >0.03. Improvement >0.05 = strong signal.

## Falsification
All combinations within ±0.03 of H559 → H559's params are globally robust.
