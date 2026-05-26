# H223 — LeverageReduction

## Hypothesis

Financial de-leveraging: long stocks with the largest YoY decrease in leverage ratio (LTD / total_assets). Declining leverage signals improving financial health, management discipline, and reduced bankruptcy risk. The academic literature on leverage and returns (Fama-French 1992, George-Hwang 2010) documents that leverage decreases predict positive future excess returns — often attributed to reduced financial distress risk and improved earnings quality.

This is distinct from CapitalDiscipline (H210) which computes an absolute change in LTD level and combines it with other signals. Here we use the normalized leverage ratio (LTD/TA), which is scale-invariant and directly comparable across companies of different sizes.

**Mechanics**: lev = LTD / total_assets. lev_change = lev - lev.shift(252). Sector-neutral bottom 20% by lev_change (most reduced leverage = buy). Monthly rebalance.

## Expected behavior by regime

- **2000-2002 tech crash**: Highly leveraged tech was being written down; de-leveraging companies (paying down debt) avoided forced selling.
- **2008-2009 GFC**: Post-crisis de-leveraging is a recovery signal. Companies forced into de-leveraging by lender pressure are different from voluntary de-leveragers.
- **2020 COVID**: Many companies drew down credit lines, increasing leverage. Those that didn't (or paid down quickly) score well.
- **Low-vol 2013-2017**: Steady cash-generative businesses naturally de-lever — consistent with quality compounders.

## Success criteria

- OOS Sharpe > 0.51 with CI lower bound > 0.
- H001 corr near zero (no price component).
- Meaningful improvement in Max DD over SPY (leverage reduction = lower distress risk).

## Falsification

If CI lower ≤ 0, voluntary de-leveraging does not predict future returns above chance. If Sharpe < 0.51, the de-leveraging signal is too slow/noisy relative to other quality metrics.

## Results (2026-05-25)

**Decision: KEEP**

| Metric | Value |
|---|---|
| OOS Sharpe | 0.716 |
| 95% CI | [0.305, 1.118] |
| CAGR | 8.31% |
| Max DD | -38.22% |
| H001 corr | -0.022 |

**Per-fold Sharpes**: 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.441, 1.993, 0.938, 1.922, 0.960, 0.853

Folds 0-5 exactly zero (EDGAR gap). Folds 6-11 average 1.185. The lowest H001 corr in Batch 5 (-0.022) — consistent with the quality factors cluster. Max DD -38.22% is significantly better than SPY. Sensitivity analysis shows 126d window produces marginally higher Sharpe (0.770) — shorter-horizon de-leveraging may be more actionable.
