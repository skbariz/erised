# H164: MaxDrawdownReversal — Buy Most Beaten-Down Stocks in Last 12 Months

## Signal
Rank S&P 500 members by their maximum drawdown over the past 252 trading days (price peak to trough, full window). Long the bottom quintile — stocks with the worst 12-month maximum drawdowns. Equal-weight, monthly rebalance.

**Why it should work:** Standard 12-1m return misses intra-year crashes when stocks partially recover. A stock can have a −5% 12-month return but a −40% drawdown within that period. The max drawdown signal specifically selects the stocks that were hardest-hit intra-year and have started recovering — this is a more sensitive detector of beaten-down conditions than period return alone. Baker & Wurgler (2006) document sentiment-driven price drops that are partially reversed as sentiment recovers.

**Difference from H160/H161:** Long-term reversal looks back 3-5 years. Max drawdown reversal looks at the worst intra-year hit — capturing faster recovery cycles and stocks with sudden drops that have already started to stabilize.

## Expected regime behavior
- **2008-2009**: High max drawdowns across the board; portfolio holds financials and cyclicals during recovery. Favorable for 2009 recovery.
- **2020 COVID**: March 2020 crash produces extreme max drawdowns in energy, airlines, hotels. Portfolio loads these up as they partially recover.
- **2013-2017**: Few extreme drawdowns — signal identifies idiosyncratic fallers (e.g., biotech failures, energy during oil price collapse 2015-2016). Mixed.

## Success criteria
- OOS Sharpe > 0.51
- 95% CI lower bound > 0
- H001 correlation expected < 0.50 (contrarian character)

## Falsification
- If max drawdown stocks are predominantly distressed names that never recover (survivorship bias test), the signal is a value trap
- If OOS Sharpe < 0.30, the signal selects damaged franchises rather than temporarily depressed ones
- The S&P 500 universe provides natural protection: members are periodically removed when they deteriorate beyond repair, so the "value trap" risk is reduced vs a broader universe

## Implementation notes
- Max drawdown: `(price / price.rolling(window).max() - 1).min()` over the formation window
- `window=252` trading days
- Rank ascending (most negative max drawdown = ranked first for longs)
- `top_quantile=0.20`
