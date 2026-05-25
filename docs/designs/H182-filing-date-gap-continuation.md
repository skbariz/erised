# H182: FilingDateGapContinuation

## Hypothesis

**Signal:** Event-driven. Detect EDGAR filing dates (when NI panel value changes) and observe the stock's price reaction on that day. If the price gap up exceeds `gap_threshold=2%`, go long for `hold_days=20` trading days.

**Why it should work:** EDGAR 10-Q/10-K filings sometimes reveal information not fully in the earnings press release (detailed segment data, cash flow details, footnotes). When a stock gaps up on or around a filing date, it suggests the market is reacting positively to incremental information disclosure. The continuation captures the post-filing price drift as the information is absorbed by slower-moving investors.

**Mechanism:** Filing detection uses `ni.diff().abs() > 0` — when the EDGAR XBRL net income value changes, this proxies for a new 10-Q or 10-K submission. Gap is the day's return on that date.

**Expected regime behavior:**
- **2000-2002 tech crash:** Few positive gaps; strategy would have near-zero exposure. Defensive.
- **2008-2009 GFC:** Similar — positive gaps rare during the downturn. Near-zero exposure.
- **2020 COVID:** Very asymmetric — tech companies filing positive results while market crashes. Interesting test.
- **Low-vol 2013-2017:** Regular filing cadence with modest positive gaps. Steady alpha.

**Data constraint:** Same EDGAR coverage gap pre-2009.

## Success criteria

- OOS Sharpe > 0.51
- 95% CI lower bound > 0
- H001 corr < 0.60 (event-driven, not trend)

## Falsification

If positive filing-date gaps do not continue over 20 days, the EDGAR filing itself contains no incremental information beyond what the earnings press release announced. The strategy degenerates to random entry. Kill.

## Implementation

- `FilingDateGapContinuation` in `src/quant/strategies/cross_sectional.py`
- Filing detection: `(ni.diff().abs() > 0) | (ni.notna() & ni.shift(1).isna())`
- Sensitivity sweep: gap_threshold ∈ {1%, 2%, 3%}

## OOS Results

*(to be filled after backtest)*
