# H179: LowBetaQuality

## Hypothesis
The intersection of low systematic risk (beta) and high fundamental quality (GP/A) identifies stocks that are doubly mispriced: overweighted by constrained investors (BAB anomaly) AND underweighted by investors chasing earnings stories (gross profitability anomaly). The composite score is (1/2)*z(-beta) + (1/2)*z(GP/A), sector-neutral, selecting the top 20% within each GICS sector.

This is the pure defensive combination — no momentum component unlike H178. The hypothesis is that beta and profitability capture orthogonal sources of mispricing that compound when combined, and that the sector-neutral constraint prevents style tilts from dominating returns.

## Expected behavior
- **2000-2002 tech crash:** Low-beta + high-GP/A strongly avoids overpriced momentum tech — best possible period
- **2008-2009 GFC:** Low-beta provides protection; GP/A holds fundamental floor; expect positive fold
- **2020 COVID:** Quality stocks initially crash with market; low-beta reduces exposure magnitude
- **Low-vol 2013-2017:** Low-beta lags in bull; GP/A compensates; should outperform H175 (pure BAB)

## Success criteria
- OOS Sharpe > 1.0 with CI lower bound > 0.5
- All 12 folds positive (beta from prices = always available; GP/A gaps only affect weight magnitude in early folds)
- Sharpe > H175 BAB (1.088) — quality adds value over beta alone

## What would falsify this
- Sharpe similar to or below H175 (1.088) — GP/A not adding to beta signal
- Large negative fold in 2008-2009 (both signals should provide protection in crashes)

## Results (OOS)
- **Decision: KEEP**
- Sharpe 1.199, CI [0.788, 1.609], CAGR 18.16%, MaxDD -40.44%
- H001 corr: +0.6441
- All 12 folds positive (min: +0.073 in 2008-2010 — remarkable protection in GFC)
- Sensitivity: quantile 10%/20%/30% Sharpe 1.284/1.199/1.119 — monotone, tighter cut is stronger
- Beats H175 BAB (+0.111 Sharpe) — GP/A adds genuine orthogonal value
- Best strategy in Batch 6; second-highest Sharpe in H150-H179 range after H168 (1.181) and H169 (1.073)
