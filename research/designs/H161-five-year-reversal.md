# H161: FiveYearReversal — 5-Year Price Reversal (Long-Horizon Contrarian)

## Signal
Same mechanism as H160 but using a 60-month formation window (months t-61 to t-13). Long the bottom quintile of 5-year return losers. Equal-weight, monthly rebalance.

**Why it should work:** De Bondt & Thaler's original paper used 3-5 year formation periods and found the longer horizon produced stronger reversal. A 5-year loser has typically experienced multiple rounds of bad news, analyst downgrades, and institutional selling — the overreaction accumulates over a longer period and the subsequent recovery is correspondingly larger. In the S&P 500 universe, 5-year losers are often restructuring stories or cyclical troughs.

## Expected regime behavior
- 60-month formation means the signal at any point in time looks back to a very different market regime, providing natural diversification vs shorter-horizon signals.
- The skip_days=21 still isolates us from short-term momentum crash after ranking.
- **2000-2002**: 5-year losers (1996-2000 formation) include stocks that never participated in the tech boom — may hold utility/staple names that held up well.
- **2008-2009**: 5-year losers (2004-2009) include commodity cyclicals and early financials; partial recovery 2009-2011.
- **Versus H160**: Longer horizon = more contrarian, more stable portfolio (slower rank turnover), potentially more diversification benefit.

## Success criteria
- OOS Sharpe > 0.51
- 95% CI lower bound > 0
- H001 correlation expected < 0.35 (longer contrarian horizon = less momentum loading)

## Falsification
- If identical to H160 within noise, the 36-month signal already captures all available reversal information
- If H161 has worse Sharpe than H160, the longer formation introduces noise at large-cap scale

## Implementation notes
- Formation window: `long_days=1260` (≈60 months), `skip_days=21`
- Rank ascending, long bottom quintile
- `top_quantile=0.20`
