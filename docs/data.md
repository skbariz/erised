# Data

Erised does not ship market data. The `erised data` commands download it from public
sources into a local Parquet store that is never committed.

## Location

The data directory resolves in this order:

1. `ERISED_DATA_DIR`, if set.
2. `<repo>/data` when running from a source checkout.
3. `./data` relative to the working directory when installed as a package.

## Layout

```
data/
├── <SYMBOL>.parquet              full daily history (erised data fetch)
├── <SYMBOL>_research.parquet     everything before the holdout start
├── <SYMBOL>_holdout.parquet      sealed holdout
├── HOLDOUT_HASH.txt              SHA-256 of each holdout file (erised data seal)
├── stocks/<TICKER>.parquet       constituent prices (erised data stocks)
├── fundamentals/<TICKER>.parquet SEC EDGAR fundamentals (erised data fundamentals)
├── sp500_members.parquet         point-in-time membership, dates by tickers
├── sp500_sectors.parquet         ticker to GICS sector
└── sp500_tickers.txt             every ticker that has been a constituent
```

Every bar file follows `erised.data.schema`: tz-naive daily `DatetimeIndex` and columns
`open, high, low, close, volume, adj_close`, validated on load.

## Sources

| Data | Source | Notes |
|---|---|---|
| ETF and index bars | Yahoo Finance via [`yfinance`](https://github.com/ranaroussi/yfinance) | Adjusted close cross-checked against Stooq when available |
| VIX | Yahoo Finance (`^VIX`) | Used as a regime signal only |
| S&P 500 membership and sectors | Wikipedia "List of S&P 500 companies" and its change log | Reconstructed history, reliable from about 2000; earlier folds may carry some survivorship bias |
| Fundamentals | [SEC EDGAR XBRL company facts](https://www.sec.gov/edgar/sec-api-documentation) | Keyed by filing date; dense coverage from 2009 |

**Terms of use are your responsibility.** Yahoo Finance data is provided for personal use
under Yahoo's terms, and redistribution is generally not permitted, so don't commit
downloaded files to a public repository. The SEC asks automated clients to identify
themselves (hence `SEC_EDGAR_USER_AGENT`) and to stay under 10 requests per second;
`erised data fundamentals` sleeps between requests to respect that. Wikipedia content is
CC BY-SA.

## Data availability

Several ETFs launched after the 1993 start of the research period: TLT, SHY, IEF, and LQD
(2002), GLD and VNQ (2004), DBC (2006), HYG (2007). Strategies that need them hold no
position in early folds, which drags down combined out-of-sample metrics. Fundamentals are
sparse before 2009.

## Synthetic data

`erised.data.synthetic.synthetic_bars()` builds schema-valid random-walk bars in memory, and
`erised data synthetic` writes a synthetic file for every symbol so the full
pipeline (sealing, loaders, runners) can be exercised offline. CI does exactly this.
Results on synthetic data mean nothing about a strategy.
