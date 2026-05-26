# H261: LowNetOperatingAssets — Balance Sheet Bloat Avoidance

## Hypothesis

Hirshleifer, Hou, Teoh, and Zhang (2004) show that Net Operating Assets (NOA) negatively
predicts future returns. High NOA reflects cumulative accounting earnings that exceeded
cumulative cash generation — investors overpay for balance-sheet bloat.

Signal: NOA proxy using VALID_FIELDS:
  operating_assets ≈ total_assets - current_liabilities (non-debt operating assets)
  operating_liabilities ≈ current_liabilities - (current_assets - operating_cf/4)  [rough LC proxy]
  NOA = (total_assets - current_liabilities) / total_assets

Simplified: NOA ≈ (total_assets - current_assets) / total_assets + (current_assets - current_liabilities) / total_assets
         = (total_assets - current_liabilities) / total_assets

Rank ascending (low NOA = low bloat = best). Sector-neutral top-quintile, monthly rebalance.

Expected regime behavior:
- **2000–2002**: Strong — tech companies had high NOA from capitalized costs.
- **2008–2009**: Moderate — banks loaded with off-balance-sheet assets; hard to proxy well.
- **2013–2017**: Moderate signal strength across the board.
- **2020**: Neutral — COVID shocks are symmetric across NOA levels.

Success criteria: OOS Sharpe ≥ 0.70, Max DD < −45%, CI lower bound > 0.
Falsification: Correlation with H166 (Low Accruals Quality) > 0.90 — not independent.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
