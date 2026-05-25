# H160: LongTermReversal — 3-Year Price Reversal (De Bondt-Thaler Contrarian)

## Signal
Rank S&P 500 members by their 36-month past total return, using months t-37 through t-13 (skipping the most recent year to avoid momentum contamination). Long the bottom quintile (biggest 3-year losers). Equal-weight, monthly rebalance.

**Why it should work:** De Bondt & Thaler (1985) documented that stocks experiencing extreme losses over 3–5 years subsequently outperform extreme winners by ~25% over the following 3 years. The mechanism is investor overreaction: bad news pushes prices below fundamental value; subsequent regression to the mean generates alpha. This is a pure price-based approximation of cheapness — a stock that has lost 60% over 3 years is likely to have a lower P/E and P/B than a stock that gained 100%.

## Expected regime behavior
- **2000-2002 tech crash**: Tech losers pile into the portfolio, but the signal fired *before* the crash (36-month winners from 1997-2000 are the contrarian shorts; the longs were the prior-era losers). Mixed.
- **2008-2009 GFC**: Cyclicals and financials were deep 3-year losers by 2009, many subsequently recovered strongly. Favorable.
- **2020 COVID**: Many energy/travel stocks had 3-year losses by early 2020 that weren't COVID-related; signal may be caught holding wrong names during initial crash, but recovery from late 2020 on.
- **2013-2017 low-vol bull market**: Utilities and defensive stocks were prior losers; energy peaked ~2014 and fell — portfolio may hold energy names too early.

## Success criteria
- OOS Sharpe > 0.51 (beats SPY benchmark)
- 95% CI lower bound > 0
- H001 correlation < 0.40 (anti-momentum character expected)
- No single fold dominates (fold Sharpe range < 3.0)

## Falsification
- If OOS Sharpe < 0.30, the large-cap universe has priced out De Bondt-Thaler reversal
- If H001 corr > 0.60, the signal is just loading on momentum in disguise
- If CI lower bound ≤ 0, the signal has too much variance across market regimes

## Implementation notes
- Formation window: `long_days=756` (≈36 months of trading days), `skip_days=21`
- Rank ascending (lowest return = highest rank for longs)
- `top_quantile=0.20` (bottom quintile of returns = top quintile for longs)
