# Contributing to Erised

Thanks for helping. Bug reports, documentation fixes, new strategies, and engine
improvements are all welcome.

## Before you start

- Search existing issues first. For anything larger than a small fix, open an issue to
  discuss the approach before writing code.
- Security problems go through [SECURITY.md](SECURITY.md), not public issues.
- By contributing you agree your work is released under the [MIT License](LICENSE), and you
  agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

```bash
git clone https://github.com/<you>/erised.git
cd erised
uv sync --all-extras
uv run pre-commit install
uv run python scripts/generate_synthetic_data.py
uv run python scripts/seal_holdout.py
```

The last two commands create synthetic data so the full test suite runs offline. They
rewrite `data/HOLDOUT_HASH.txt`; don't commit that change
(`git restore data/HOLDOUT_HASH.txt`).

## Checks

Pull requests must pass the same checks CI runs:

```bash
uv run ruff check src scripts tests examples
uv run ruff format --check src scripts tests examples
uv run pytest -q
```

`research/` is a frozen archive and is excluded from linting. Don't reformat it.

## Adding a strategy

1. Write down the hypothesis, expected behavior in key regimes, and what would falsify it
   before you run anything. A short section in the pull request is enough; see
   `research/designs/` for examples.
2. Implement the class in `src/erised/strategies/base.py` or `cross_sectional.py`, matching
   the style of nearby classes. See [Writing strategies](docs/writing-strategies.md).
3. Add tests: a lookahead check (`assert_no_lookahead` or `assert_no_lookahead_multi`) plus
   behavioral tests on synthetic data.
4. If you report performance, include the out-of-sample Sharpe, its 95% confidence
   interval, max drawdown, and per-fold Sharpe, and state the data range you used. Don't
   tune parameters on out-of-sample results.

## Pull requests

- Keep each pull request focused on one change.
- Describe what changed and how you tested it.
- Never commit market data, credentials, `.env` files, or personal paths. Pre-commit runs a
  secret scanner; don't bypass it.
