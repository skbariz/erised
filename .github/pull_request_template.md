## Summary

<!-- What changed and why. -->

## Testing

<!-- Commands you ran and what they showed. -->

## Checklist

- [ ] `uv run ruff check src scripts tests examples` and `ruff format --check` pass
- [ ] `uv run pytest -q` passes
- [ ] New strategies include a lookahead test
- [ ] Any reported performance includes OOS Sharpe, 95% CI, max drawdown, and per-fold Sharpe
- [ ] No data files, credentials, `.env`, or machine-specific paths are committed
