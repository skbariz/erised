# H224 — PiotroskiFScore

## Hypothesis

The Piotroski (2000) F-Score is a 9-component binary composite of profitability, leverage/liquidity, and operating efficiency signals. It was designed to distinguish financially strong value stocks from value traps among high book-to-market firms, but has been shown to work broadly. Here we apply it in a sector-neutral long-only context: rank all S&P 500 stocks by F-score and long the top 20% within each sector.

**9 components:**
- Profitability (F1-F4): ROA > 0, OCF > 0, ROA improving YoY, OCF > NI (accruals quality)
- Leverage/Liquidity (F5-F7): LTD/TA decreasing, current ratio improving, no share dilution
- Efficiency (F8-F9): gross margin improving, asset turnover improving

**Mechanics**: All 9 signals computed on 252-day YoY basis. F-score = sum (0-9). Sector-neutral top 20% by F-score. Monthly rebalance.

## Expected behavior by regime

- **2000-2002 tech crash**: Most tech stocks fail F1 (ROA < 0) and F2 (OCF < 0) → low F-scores → excluded. Strategy avoids crash epicenter.
- **2008-2009 GFC**: Financial stocks fail multiple leverage/liquidity components → excluded.
- **2020 COVID**: Broad F-score deterioration; post-COVID recovery favors companies that maintained quality.
- **Low-vol 2013-2017**: Compounders with steady improvements across all 9 components selected — ideal regime.

## Success criteria

- OOS Sharpe > 0.51 with CI lower bound > 0.
- H001 corr near zero (9-component fundamental composite has no price component).
- Folds 6-11 average Sharpe > 1.0 in EDGAR-covered period.

## Falsification

If CI lower ≤ 0, the aggregate F-score does not discriminate winners from losers in the S&P 500 (unlike the small-cap/value universe it was designed for). The S&P 500 is already a quality-filtered universe; F-score may not add much within it.

## Results (2026-05-25)

**Decision: KEEP**

| Metric | Value |
|---|---|
| OOS Sharpe | 0.752 |
| 95% CI | [0.346, 1.162] |
| CAGR | 8.53% |
| Max DD | -37.02% |
| H001 corr | -0.022 |

**Per-fold Sharpes**: 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.642, 2.018, 0.867, 1.748, 1.033, 0.950

Folds 0-5 exactly zero (EDGAR gap). Folds 6-11 average 1.210. H001 corr -0.022 (tied with H223 for lowest in batch). Max DD -37.02% is the second-best in batch. The sector-neutral application of F-score works within the S&P 500, though the Sharpe is moderate (0.752) due to the coverage gap. The F-score is a successful replica of Piotroski (2000) in our framework.
