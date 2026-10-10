# H178: DefensiveQualityMomentum

## Hypothesis
Combining quality (defensive: high GP/A + low beta) with price momentum in a sector-neutral composite should outperform either factor alone. Quality provides the fundamental anchor (high-profitability, low-beta stocks are mispriced relative to fair value); momentum provides the timing signal (buy quality stocks already trending up; avoid quality stocks in a downtrend). The composite score is (1/2)*z(quality) + (1/2)*z(momentum) where quality = (z(GP/A) + z(-beta))/2.

This combines Novy-Marx (2013) gross profitability and Frazzini-Pedersen (2014) BAB within the momentum selection framework. Beta is derived from prices (always available); GP/A from EDGAR (available from ~2009, zero weights in earlier folds).

## Expected behavior
- **2000-2002 tech crash:** Both quality and momentum filters favor non-tech stocks — should be excellent period
- **2008-2009 GFC:** Momentum filter catches the downturn; quality provides some cushion; expect near-zero or slightly negative fold
- **2020 COVID:** Mixed; low-beta quality stocks hold up but momentum signal may flip
- **Low-vol 2013-2017:** Steady; quality + momentum both work well in calm bull markets

## Success criteria
- OOS Sharpe > 1.0 with CI lower bound > 0.5 (both quality and momentum are well-established)
- H001 corr in 0.55–0.75 range
- Meaningful alpha vs plain momentum H150 (0.981)

## What would falsify this
- Composite underperforms both quality alone and momentum alone — combination not additive
- Large negative fold in 2000-2002 (quality + momentum both long tech in 1999 = disaster)

## Results (OOS)
- **Decision: KEEP**
- Sharpe 1.125, CI [0.721, 1.539], CAGR 19.11%, MaxDD -46.95%
- H001 corr: +0.6644
- 11/12 folds positive (fold 5 = -0.092 in 2008-2010 — momentum crash)
- Sensitivity: quantile 10%/20%/30% Sharpe 1.193/1.125/1.075 — tighter cut is stronger
- Beats H150 baseline by +0.144 Sharpe
