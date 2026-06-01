# H450 — UltimateH401to450MDR

**Date:** 2026-05-31
**Family:** Set6 terminal ensemble
**Plan doc:** docs/designs/B401-450.md

## Hypothesis

SLSQP maximum diversification ratio (MDR) ensemble over the complete Set 6 pool:
14 H300/H350 core ETF strategies + 42 kept CS strategies from H401–H448.

H449 used 14 ETF core + 17 CS (selected subset). H450 adds the full 25 remaining kept CS
strategies from batches 7, 8, 9, and 10 (H431–H448, minus H440 which was killed).

**What does this test?** Whether the broader Set 6 CS strategies (credit risk, ETF crowding,
institutional horizon proxies, meta-composites) add MDR diversification beyond H449's subset.

## Expected regime behavior

- All-fold CS stars (H447 CreditQualityMomentumCombo, H412, H415, H429) will get MDR weight
- EDGAR-gap strategies (folds 0-5 zero) will receive near-zero MDR weight — optimizer can't use them
- Net result: likely very similar to H449 (Sharpe 1.164), maybe slightly better from H447 all-fold contribution

## Success criteria

- Sharpe > 0.51, CI lower > 0 (minimum keep bar)
- Any improvement over H449 (Sharpe 1.164) is a bonus
- Does not need to beat H300/H350 (1.525) — that frontier is confirmed closed

## What would falsify

- Sharpe below H449 (would mean adding more strategies hurts through overfitting to the optimizer)
- MDR assigns zero weight to all new additions (confirms H449 already captured max value)
