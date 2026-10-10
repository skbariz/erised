# H277: CoskewnessPortfolioPenalty — Avoid High Coskewness with Market

## Hypothesis

Harvey and Siddique (2000) define coskewness as E[(ri−μi)(rm−μm)²] normalized by
(σi × σm²). Stocks with high positive coskewness perform poorly when the market is crashing
(they crash harder during market down-moves). Investors rationally require a premium for
holding high-coskewness names. We test whether avoiding high-coskewness names (loading on
low-coskewness stocks, which provide diversification value) delivers alpha.

Signal: Compute rolling 252d coskewness = mean((ri-μi)(rm-μm)²) / (std(ri) × std(rm)²)
using daily returns and SPY as the market. Rank ascending (most negative coskewness = best,
provides diversification). Long top-quintile, sector-neutral, monthly rebalance.

Distinction from H276 (NegativeSkewnessAvoidance): H276 uses marginal skewness of each
stock in isolation. H277 uses coskewness with the market — a portfolio-level metric capturing
how much each stock exacerbates market crashes.

Expected regime behavior:
- **2000–2002**: High-coskewness names (amplify market crashes) are punished. Avoiding them
  should protect capital during 2000-2002 decline.
- **2008–2009**: Signal most informative in crashes — high-coskewness stocks crash harder.
- **2013–2017**: Low signal in trending market; coskewness stable across names.
- **2020**: High-coskewness names crash harder in Feb-Mar 2020; signal fires correctly.

Success criteria: OOS Sharpe ≥ 0.70, CI lower > 0, H276 corr < 0.80.
Falsification: H276 corr > 0.85 — coskewness and marginal skewness are interchangeable measures.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.936 |
| 95% CI | [0.526, 1.345] |
| CAGR | 18.49% |
| Max DD | -54.83% |
| H001 corr | -0.001 |
| Decision | KEEP |
