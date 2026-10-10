# Getting started

## 1. Install

Erised needs Python 3.11 or newer. The project uses [uv](https://docs.astral.sh/uv/) for
environments and locking.

```bash
git clone https://github.com/skbariz/erised.git
cd erised
uv sync --all-extras        # core + research + live extras, plus dev tools
```

Extras:

| Extra | Adds | Needed for |
|---|---|---|
| *(core)* | numpy, pandas, pyarrow, scipy, scikit-learn, requests, yfinance | the library |
| `research` | matplotlib, lxml | charts, S&P 500 membership scrape |
| `live` | alpaca-py | `erised.execution.paper_trader` |

If you prefer pip: `pip install -e ".[research]"`.

## 2. Try it offline

Every example runs on synthetic prices, so you can explore the API before downloading
anything:

```bash
uv run python examples/01_quickstart.py
uv run python examples/02_custom_strategy.py
uv run python examples/03_multi_asset_risk_parity.py
```

## 3. Fetch real data

```bash
uv run python scripts/fetch_data.py
```

This downloads daily bars for SPY and the ETFs used by the multi-asset strategies from Yahoo
Finance, cross-checks adjusted closes against Stooq, and writes `data/<SYMBOL>.parquet`.
Edit `SYMBOLS` in the script to change the universe. Data is stored in `./data` by default;
set `ERISED_DATA_DIR` to put it elsewhere (see [Data](data.md)).

Cross-sectional strategies also need constituent data:

```bash
uv run python scripts/fetch_sp500_members.py      # point-in-time S&P 500 membership
uv run python scripts/fetch_stocks.py             # one Parquet per historical constituent
SEC_EDGAR_USER_AGENT="your-project you@example.com" \
  uv run python scripts/fetch_fundamentals.py     # SEC EDGAR XBRL fundamentals
```

The SEC requires every client to identify itself with a contact address, so
`fetch_fundamentals.py` will not run without `SEC_EDGAR_USER_AGENT`.

## 4. Seal the holdout (once)

```bash
uv run python scripts/seal_holdout.py
```

This splits each series at 2023-01-01 into `<SYMBOL>_research.parquet` and
`<SYMBOL>_holdout.parquet` and records SHA-256 hashes in `data/HOLDOUT_HASH.txt`. Do all
research with `load_research()`. `load_holdout()` refuses to run unless you pass
`_i_understand_this_is_final_evaluation=True`; spend it once, at the end.

The committed `HOLDOUT_HASH.txt` records the seal used for the published research. Your own
seal will produce different hashes because your download date differs; that is expected.

## 5. Verify

```bash
uv run pytest -q
uv run python scripts/sanity_check.py    # MA(50,200) walk-forward report on your data
```

## 6. Next steps

- Read [Methodology](methodology.md) for the conventions every strategy must follow.
- Follow [Writing strategies](writing-strategies.md) to add your own.
- Browse [`research/`](../research/README.md) to see how 670 hypotheses were run and recorded.

## Optional: paper trading

`erised.execution.paper_trader` is a thin Alpaca wrapper that can turn a strategy's latest
target into an order. Copy `.env.example` to `.env`, fill in **paper** keys from
[alpaca.markets](https://alpaca.markets/), and export them into your shell. Never commit
`.env`; it is git-ignored.
