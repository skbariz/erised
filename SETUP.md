# Setup

## 1. Unzip into a folder

```bash
unzip quant.zip
cd quant
```

## 2. Install uv (if you don't have it)

uv is a fast Python package manager. One of:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

## 3. Create the environment and install dependencies

```bash
uv sync --all-extras
```

This creates `.venv/` and installs everything in `pyproject.toml`.

## 4. Fetch real SPY data

```bash
uv run python scripts/fetch_data.py
```

This pulls SPY history from yfinance and cross-validates against Stooq.
Takes ~10 seconds. Writes `data/SPY.parquet`.

## 5. Seal the holdout (do this ONCE, ever)

```bash
uv run python scripts/seal_holdout.py
```

This splits the data into research (1993–2022) and holdout (2023–present).
Writes `data/SPY_research.parquet`, `data/SPY_holdout.parquet`, and
`data/HOLDOUT_HASH.txt`. The hash file gets committed to git.

## 6. Verify everything works

```bash
uv run pytest -v
uv run python scripts/sanity_check.py
```

You should see 21 tests pass and a walk-forward report for MA(50,200)
against real SPY data.

## 7. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Phase 1 platform"
git branch -M main
git remote add origin git@github.com:<your-handle>/quant.git
git push -u origin main
```

## 8. (Later) Set up Alpaca paper trading

When ready for Phase 2 deployment, get free API keys from
https://alpaca.markets/ (paper account) and set:

```bash
export ALPACA_API_KEY=...
export ALPACA_SECRET_KEY=...
```

These should NEVER be committed to git. Put them in a `.env` file
(already in `.gitignore`) or set them in your shell.

## Optional: install pre-commit hooks

```bash
uv run pre-commit install
```

This runs ruff and a few sanity checks before every commit.
