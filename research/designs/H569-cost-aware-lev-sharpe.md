# H569: CostAwareLevSharpe

## Signal
Replace flat 0.5% annual borrow cost with historical step-function Fed Funds Rate + 50bps
spread. Same lev-Sharpe SLSQP objective and MDR warm-start as H559. Leverage params:
6.5%/2.0x/20d. This is a robustness test — H559 was developed assuming 0.5% annual
borrow cost throughout 1993-2022, which was accurate 2009-2022 but understates cost in
1993-2000 (Fed Funds ~5.5%) and 2004-2007 (~5.0%).

## Why should it work?
H559's borrow cost assumption (0.5%/yr flat) was approximately correct post-2009 but
substantially wrong pre-2008. The realistic pre-2008 cost is ~5-10x higher. If the
portfolio's alpha comes disproportionately from the low-rate era, H559's Sharpe may be
overstated. Conversely, if H559 is robust to realistic costs, the 1.879 result holds.
The hypothesis also tests whether the optimizer re-allocates when pre-2008 costs are
properly penalized — strategies that are leverage-insensitive (e.g. H473 L/S) may gain
weight vs rate-sensitive ETFs.

## Expected behavior in key regimes
- 1993-2000 (5.5% borrow): significant de-leveraging of carry-sensitive strategies.
  Leverage periods >1.0x are expensive. Expect fold 0-1 Sharpe to fall.
- 2000-2003 (1-4.5% borrow): moderate cost; depends on leverage levels.
- 2009-2022 (0.25-0.5% borrow): costs near H559 flat assumption → essentially unchanged.
- 2022+ (4% borrow): recent folds will show materially higher costs → lower fold Sharpe.

## Success criteria
- KEEP if Sh_lev > 1.70 (wide tolerance — primary goal is robustness assessment)
- Key metric: Sh_lev difference vs H559 (1.879). If < -0.1, H559 overstates due to cheap
  capital assumption.

## Falsification
- If Sh_lev < 1.60: historical leverage is substantially cheaper than realistic costs;
  H559 is overstated; Set 8 leverage ensemble may not hold up in live trading.
- If Sh_lev ≈ H559 (within ±0.05): borrow cost assumption is benign; H559 robust.

## Results (fill in after running)
- Decision:
- Sh_lev:
- CI:
- CAGR_lev:
- DD_lev:
- Cal_lev:
- Delta_Sharpe vs H559:
- Notes (borrow cost impact assessment):
