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
| OOS Sharpe | 0.889 |
| 95% CI | [0.479, 1.301] |
| CAGR | 11.05% |
| Max DD | −38.66% |
| H001 corr | −0.017 |
| Decision | **KEEP** |

**Sensitivity (within_sector_quantile):** 10% → 0.895 | 20% → 0.889 | 30% → 0.877. Stable across cutoffs — structural signal. Strongest standalone in batch 3.

**H261 is the batch 3 standalone champion** at 0.889. NOA proxy (TA - CL)/TA is a clean, simple accounting signal. Fold 7 (2012–2014) Sharpe +2.371 is the highest single-fold result in this batch. Near-zero H001 corr confirms orthogonality.
