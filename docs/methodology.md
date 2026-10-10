# Methodology

Erised's defaults encode one research protocol. You can change any of them, but the
framework is built to make the honest path the easy one.

## Timing convention

A strategy returns a **target position for every date**. The position decided with data
through the close of day *t* is held over *(t, t+1]* and earns day *t+1*'s return.

- `run_backtest`, `run_multiasset_backtest`, and `run_panel_backtest` apply this one-bar
  shift themselves. Do **not** shift positions again inside a strategy.
- Rolling features computed from data up to and including *t* are fine. Using anything
  from *t+1* or later (`.shift(-1)`, centered windows, full-sample normalization) is
  lookahead.
- `assert_no_lookahead(strategy_fn, prices)` truncates the data at random dates and checks
  the position on each date doesn't change when future bars are added. Run it in a test for
  every new strategy.

Positions are fractions of capital in `[-1, 1]`. The engines reject larger magnitudes;
model leverage explicitly on the return stream if you need it.

## Costs

Default cost is **0.5 bp commission + 1.0 bp slippage per unit of turnover** (`|Δposition|`),
charged in the period the new position becomes active. The cross-sectional walk-forward
defaults to 2.0 bp slippage. Override per call with `commission_bps` and `slippage_bps`.
Short borrow, financing, and market impact beyond slippage are not modeled.

## Walk-forward validation

`make_folds(prices, train_years=5, test_years=2)` builds rolling, non-overlapping test
windows. With research data from 1993 through 2022 that gives 12 folds whose test windows
cover 1998-2022.

For each fold, `walk_forward` calls `strategy_factory(train_df)` so a strategy can fit
parameters on the training window only, then generates positions over warm-up plus test
data and keeps the test slice. Stitching the test slices gives one out-of-sample return
series (`combined_returns`) and a report (`combined_report`).

Look at per-fold Sharpe (`fold_results`), not only the combined number. A strategy whose
edge comes from one or two folds is fragile.

## Keep/kill rule

The research program kept a strategy only if both of these held on out-of-sample returns:

1. The lower bound of the 95% bootstrap confidence interval on Sharpe
   (`bootstrap_sharpe_ci`) is above zero.
2. Sharpe beats SPY buy-and-hold over the same windows (0.51 for 1998-2022).

`bootstrap_sharpe_ci` uses i.i.d. resampling, which understates uncertainty for
autocorrelated returns. Treat the interval as a floor on how uncertain you should be.

## Sealed holdout

`erised data seal` writes the final years of each series (2023 onward by default) to
a separate file and records its SHA-256. `load_research()` never returns holdout rows, and
`load_holdout()` raises unless called with `_i_understand_this_is_final_evaluation=True`.
Evaluate on the holdout once, after every research decision is final. Once you've looked,
it's in-sample.

## Bias checklist

Run through this before trusting any new result:

- **Lookahead.** Are all signals computable at the decision time? Any negative shifts,
  same-bar fills, or full-sample statistics?
- **Survivorship.** For stock selection, are you using point-in-time index membership
  (`load_sp500_members`) rather than today's constituents?
- **Reporting lag.** Are fundamentals keyed to their filing date (`load_pit_field`), not the
  fiscal period end?
- **Data availability.** Does the strategy depend on an asset that didn't exist for early
  folds (TLT 2002, GLD 2004, DBC 2006, HYG 2007)? Zero-exposure folds dilute results.
- **Parameter snooping.** Were parameters fixed before looking at out-of-sample results?
  Record sensitivity sweeps, but don't re-pick the canonical value from them.
- **Selection.** Ensembles built from strategies chosen *because* they did well
  out-of-sample, or weighted by an optimizer run on those same returns, are in-sample at the
  portfolio level.
- **Multiple testing.** With many hypotheses, some will pass by chance. Hold the bar higher
  as the count grows, and lean on the holdout.
