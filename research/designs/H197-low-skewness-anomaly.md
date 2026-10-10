# H197: LowSkewnessAnomaly

## Hypothesis
Bali, Cakici, and Whitelaw (2011) document that investors overpay for lottery-like stocks with high positive return skewness. The demand for positively skewed assets drives their prices above fundamental value, leading to subsequent underperformance. Conversely, low-skewness (negatively skewed) stocks are underpriced and outperform. Signal = -rolling_skewness(252d). Sector-neutral top 20%, monthly rebalance.

## Expected regime behavior
- **2000-2002 tech crash**: Tech stocks had extreme positive skewness pre-crash; signal avoids them
- **2008-2009 GFC**: Lottery demand often spikes in crises (investors chase rebounds); signal avoids spike-seekers
- **2013-2017 low-vol**: Lottery demand premium should be stable; consistent signal in calm periods
- **2020 COVID**: Meme-like stocks may have extreme positive skew; signal exits these

## Success criteria
- OOS Sharpe > 0.51
- CI lower bound > 0
- H001 corr < 0.70 (skewness signal should be somewhat decorrelated from momentum)

## Falsification
- If 252-day rolling skewness is too noisy for this universe, signal variance kills the OOS Sharpe
- If the skewness premium is already arbitraged in large-cap S&P 500 stocks, no edge expected

## Parameters
- `skew_window`: 252 days (canonical)
- `within_sector_quantile`: 0.20
- `rebalance_period`: 21 days

## Results (OOS 1993-2021, 12 folds)
- **Decision: KEEP**
- OOS Sharpe: 0.879  CI [0.477, 1.287]
- CAGR: 16.97%  Max DD: -51.84%
- H001 corr: +0.612 (lower than most momentum strategies)
- Fold Sharpes: 0.92, 1.43, 0.83, 1.56, 1.19, 0.24, 0.66, 1.98, 0.65, 2.08, 1.05, 0.90
- Sensitivity: 126d skew_window best (0.959) — shorter window captures more recent lottery demand
- H001 corr of +0.612 is notable — lower than H195/H196 (~0.69), consistent with different signal type
