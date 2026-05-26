# H221 — ROEMomentum

## Hypothesis

YoY improvement in Return on Equity (NI / book_equity). ROE improvement signals a company becoming more profitable relative to its equity base — either through earnings growth, capital discipline, or leverage efficiency. The academic literature (Fama-French 1995, Chen, Novy-Marx 2013) consistently finds that profitability trends predict cross-sectional returns. This tests the ROE *change* signal specifically: companies whose earnings power is accelerating relative to equity should outperform.

**Mechanics**: roe = NI / book_equity (filtered to positive book_equity). roe_change = roe - roe.shift(252). Sector-neutral top 20% by YoY ROE improvement. Monthly rebalance.

## Expected behavior by regime

- **2000-2002 tech crash**: Tech companies with negative NI had undefined/negative ROE — excluded. ROE-improving companies tended to be defensive and industrial sectors.
- **2008-2009 GFC**: Widespread ROE collapse; post-2009 recovery selects the companies that restored profitability fastest.
- **2020 COVID**: Short-term NI disruption; ROE improvement in folds 6-11 reflects the 2010-2022 recovery cycle.
- **Low-vol 2013-2017**: Steady compounders with consistent ROE improvement selected — good regime for this signal.

## Success criteria

- OOS Sharpe > 0.51 with CI lower bound > 0.
- H001 corr near zero (no price component — decorrelated from trend-following).
- Folds 6-11 average Sharpe > 1.0 (underlying signal strength in EDGAR-covered period).

## Falsification

If CI lower ≤ 0 or Sharpe < 0.51, ROE improvement does not predict cross-sectional returns beyond chance. If folds 6-11 average is weak, the signal is noise even in the covered period.

## Results (2026-05-25)

**Decision: KEEP**

| Metric | Value |
|---|---|
| OOS Sharpe | 0.717 |
| 95% CI | [0.309, 1.135] |
| CAGR | 8.41% |
| Max DD | -38.34% |
| H001 corr | -0.020 |

**Per-fold Sharpes**: 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.374, 1.924, 0.786, 1.906, 1.068, 0.949

Folds 0-5 exactly zero (EDGAR coverage gap). Folds 6-11 average 1.168 — strong ROE improvement signal in the EDGAR-covered period. H001 corr -0.020 confirms decorrelation from equity trend. Max DD -38.34% is substantially better than SPY (-55.92%). Overall Sharpe 0.717 is diluted by 6 zero folds; true signal quality is higher.
