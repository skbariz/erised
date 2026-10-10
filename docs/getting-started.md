# Getting started

## 1. Install

Erised needs Python 3.11 or newer. The project uses [uv](https://docs.astral.sh/uv/) for
environments and locking.

```bash
git clone https://github.com/skbariz/erised.git
cd erised
uv sync --all-extras        # library, CLI, all extras, and dev tools
uv run erised --version
```

Extras:

| Extra | Adds | Needed for |
|---|---|---|
| *(core)* | numpy, pandas, pyarrow, scipy, scikit-learn, requests, yfinance | the library and CLI |
| `research` | matplotlib, lxml | charts, S&P 500 membership scrape |
| `live` | alpaca-py | `erised.execution.paper_trader` |

With pip instead: `pip install -e ".[research]"`, then run `erised ...` directly.

## 2. Try it offline

Every example runs on synthetic prices, so you can explore the API before downloading
anything:

```bash
uv run python examples/01_quickstart.py
uv run python examples/02_custom_strategy.py
uv run python examples/03_multi_asset_risk_parity.py
```

To exercise the full data pipeline offline, write a synthetic universe and seal it:

```bash
uv run erised data synthetic
uv run erised data seal
uv run erised backtest MovingAverageCrossover
```

Delete the data directory (or `erised data fetch` over it) before doing real research;
results on synthetic data mean nothing.

## 3. Fetch real data

```bash
uv run erised data fetch            # ETF universe + VIX (or: erised data fetch SPY TLT)
```

This downloads daily bars from Yahoo Finance, cross-checks adjusted closes against Stooq,
and writes `<SYMBOL>.parquet` files. The universe is defined in `erised.data.universe`.
Data goes to `./data` in a source checkout; set `ERISED_DATA_DIR` to put it elsewhere (see
[Data](data.md)).

Cross-sectional strategies also need constituent data:

```bash
uv run erised data sp500            # point-in-time S&P 500 membership and sectors
uv run erised data stocks           # prices for every historical constituent (resumable)
SEC_EDGAR_USER_AGENT="your-project you@example.com" \
  uv run erised data fundamentals   # SEC EDGAR XBRL fundamentals (resumable)
```

The SEC requires every client to identify itself with a contact address, so the
fundamentals download will not run without `SEC_EDGAR_USER_AGENT` (or `--user-agent`).

## 4. Seal the holdout (once)

```bash
uv run erised data seal
```

This splits each series at 2023-01-01 into `<SYMBOL>_research.parquet` and
`<SYMBOL>_holdout.parquet` and writes SHA-256 hashes to `HOLDOUT_HASH.txt` in the data
directory. Existing holdouts are never overwritten without `--force`.

Do all research with `load_research()`. `load_holdout()` refuses to run unless you pass
`_i_understand_this_is_final_evaluation=True`; spend it once, at the end. The seal used for
the published research is recorded in `research/holdout-seal.txt`. Your hashes will differ
because your download date differs.

## 5. Run a backtest

```bash
uv run erised strategies                                  # list single/multi-asset strategies
uv run erised strategies --cross-sectional                # list stock-selection strategies
uv run erised backtest MovingAverageCrossover --param fast=50 --param slow=200
uv run erised backtest LowVolTargetPortfolio --assets TLT --param vol_target=0.08
```

`erised backtest` runs a walk-forward (5-year train, 2-year test by default) on research
data and prints per-fold Sharpe, the combined report, a bootstrap confidence interval, and
SPY buy-and-hold over the same windows. Multi-asset strategies need their extra price columns
joined with `--assets` (for example `--assets TLT,GLD,VIX`).

## 6. Next steps

- Read [Methodology](methodology.md) for the conventions every strategy must follow.
- Follow [Writing strategies](writing-strategies.md) to add your own.
- Browse [`research/`](../research/README.md) to see how 670 hypotheses were run and recorded.

## Optional: paper trading

`erised.execution.paper_trader` is a thin Alpaca wrapper that can turn a strategy's latest
target into an order. Copy `.env.example` to `.env`, fill in **paper** keys from
[alpaca.markets](https://alpaca.markets/), and export them into your shell. Never commit
`.env`; it is git-ignored.
