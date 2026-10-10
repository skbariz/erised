# Research findings

Erised was built during a research program that tested 670 hypotheses (H001-H670) on daily
US data. Every runner, design doc, and result is in [`research/`](../research/README.md).
This page summarizes what the program found and, just as important, which numbers deserve
skepticism.

All figures are walk-forward out-of-sample over 1998-2022 (12 folds, 5-year train, 2-year
test, 1.5 bp cost per unit of turnover) unless stated otherwise. SPY buy-and-hold
over the same windows: Sharpe 0.51, CAGR 8.35%, max drawdown -55.9%.

## Results that hold up

These are single strategies with fixed, pre-registered parameters, so their walk-forward
numbers are genuinely out-of-sample.

| ID | Strategy | Sharpe | 95% CI | CAGR | Max DD |
|---|---|---|---|---|---|
| H001 | MA(50,200) crossover, long/flat | 0.61 | [0.18, 1.02] | 7.9% | -33.7% |
| H021 | SPY/TLT inverse-volatility risk parity | 1.03 | [0.62, 1.44] | 7.7% | -18.4% |
| H070 | Risk parity with 10% portfolio volatility target | 1.07 | [0.66, 1.48] | 7.2% | -12.3% |
| H086 | Risk parity with 8% portfolio volatility target | 1.11 | [0.71, 1.52] | 7.0% | -9.9% |
| H087 | Risk parity with drawdown-proportional de-risking | 1.01 | [0.61, 1.43] | 5.9% | -10.9% |
| H179 | Low-beta + gross-profitability S&P 500 stock selection | 1.20 | [0.79, 1.61] | 18.2% | -40.4% |
| H192 | PCA-residual Ornstein-Uhlenbeck mean reversion | 1.17 | [0.78, 1.59] | 27.5% | -50.7% |

**Sealed holdout (2023-01 to 2026-05, evaluated once):** the four-strategy H049 ensemble
returned Sharpe 1.35 with a -9.9% max drawdown, against SPY's 1.47 and -18.8%. It trailed
the index on risk-adjusted return in a strong bull market while cutting drawdown roughly in
half. This is the only fully untouched test in the program.

## Lessons by strategy family

- **Risk parity is the foundation.** Inverse-volatility SPY/TLT (H021) beat every
  single-signal timing rule, and most later improvements are overlays on it.
- **Position sizing beats signal timing.** Volatility targeting and drawdown-based
  de-risking (H070, H086, H087) improved Sharpe *and* drawdown at the same time; most timing
  signals traded one for the other.
- **Long/short on a single index fails.** Every long/short variant of a trend signal did
  worse than long/flat: shorting added whipsaw and cost without enough bear-market gain.
- **Trend signals are one signal.** MA crossovers, time-series momentum, and dual momentum
  correlate above 0.7 with each other. Adding more of them adds nothing.
- **Calendar effects are real but small.** Sell-in-May, January barometer, and Halloween
  variants passed the bar with low correlation to trend, which made them valuable ensemble
  components rather than standalone strategies.
- **Contrarian timing on the index doesn't trade enough.** Crash-buying rules fired 6-18
  times in 25 years, too few for statistical confidence.
- **Late-launching assets hurt.** Strategies needing DBC (2006) or HYG (2007) sat flat for
  3-5 of the 12 folds and almost always failed the bar.
- **Cross-sectional factors work but don't protect.** Quality, low-beta, and short-term
  reversal stock selection reached Sharpe 1.0-1.2, with drawdowns of -40% to -55%.
  Market-neutral long/short versions mostly collapsed once borrow and turnover costs were
  charged.

## Numbers to treat with caution

**Optimized ensembles.** From H095 onward, ensemble weights (maximum diversification,
max-Sharpe, Calmar objectives, and so on) were solved on the full stitched out-of-sample
return matrix and then scored on that same matrix. The components were also pre-selected
for having passed the keep bar. Their reported Sharpe ratios (1.13 for H095, 1.35 for H200,
1.53 for H300, about 2.0 for the leveraged H550-H670 sets) are therefore **in-sample at the
portfolio level**. To make them honest, fit weights on each fold's training window only and
re-score.

**Regime-threshold tuning.** Sets 8-10 tuned VIX thresholds and windows on the same data.
The program's own checks found the limit: H640 reached Sharpe 3.04 by drifting to a
parameter-grid corner, and adding a realistic one-day execution lag removed about 0.5 of
that. Prefer the lag-aware results (H646 onward) and interior parameter choices.

**Multiple testing.** With 670 hypotheses and a 95% confidence rule, dozens of "keeps" are
expected by chance alone. About 470 strategies were kept under the program's rule; many are
near-duplicates of each other. The keep label means "passed a screen," not "has an edge."

**Synthetic results.** CI runs everything on synthetic data. Any result file regenerated
in CI, or on your machine without real data, is meaningless.

## Reproducing a result

```bash
uv run erised data fetch
uv run erised data seal
uv run python research/runners/run_h021.py
```

Your numbers will differ slightly from the archive because data vendors revise history and
your download extends to a different end date. The runner overwrites
`research/results/<id>-<name>/summary.json`; use `git diff` to compare against the archived
result.
