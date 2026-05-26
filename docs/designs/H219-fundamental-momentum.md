# H219 — FundamentalMomentum

## Hypothesis

Pure accounting momentum: rank stocks by a composite of YoY growth in net income, gross profit, and operating cash flow, with no price signal at all. If fundamental growth momentum predicts future price returns, this strategy isolates that effect independently from price momentum. The hypothesis draws on the "fundamental momentum" literature (Novy-Marx 2015 et al.) that accounting growth trends predict cross-sectional returns.

**Composite score**: z(NI_growth) + z(GP_growth) + z(OCF_growth), where each growth is YoY (252-day shift). Sector-neutral top 20%, monthly rebalance.

## Expected behavior by regime

- **2000-2002 tech crash**: Fundamental momentum selects companies with growing earnings, avoiding the speculative tech stocks that had zero or negative NI. Regime-resilient.
- **2008-2009 GFC**: Earnings collapse across most sectors → all scores near zero → near-random selection. Returns to 0 during crisis periods.
- **2020 COVID**: Folds 0-5 (pre-2010) return 0 due to EDGAR coverage gap. Post-2010 coverage covers 2020 and recovery.
- **Low-vol 2013-2017**: Selects steady compounders with growing revenue and margins — consistent signal in trending regime.

## Success criteria

- OOS Sharpe > 0.51 with CI lower bound > 0.
- H001 corr near zero (no price component — pure fundamental signal should decorrelate from trend-following).
- Meaningful Sharpe in folds 6-11 (post-2010 EDGAR coverage).

## Falsification

If CI lower ≤ 0 or Sharpe < 0.51, YoY fundamental growth does not predict cross-sectional returns beyond chance. If folds 6-11 average Sharpe is weak, the fundamental signal is noise.

## Results (2026-05-25)

**Decision: KEEP** (weak — 6 zero folds, H001 corr negative)

| Metric | Value |
|---|---|
| OOS Sharpe | 0.729 |
| 95% CI | [0.329, 1.139] |
| CAGR | 8.56% |
| Max DD | -39.05% |
| H001 corr | -0.021 |

**Per-fold Sharpes**: 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.395, 2.152, 0.844, 1.670, 1.060, 0.964

Folds 0-5 exactly zero (EDGAR coverage gap 1998-2010). Folds 6-11 average 1.181 — a strong signal in the post-EDGAR period. Max DD -39.05% is best in the batch: pure fundamental selection avoids the speculative stocks that crash hardest. Negative H001 corr (-0.021) — the program's second most decorrelated strategy from equity trend after H078. **High ensemble value** despite low overall Sharpe (diluted by 6 zero folds).
