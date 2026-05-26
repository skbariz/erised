# H240: ROAMomentum — YoY Improvement in Return on Assets

## Hypothesis

Return on Assets (ROA = NI / avg_total_assets) is the DuPont composite of net profit margin and asset turnover. Improving ROA signals simultaneously improving profitability efficiency and/or asset productivity. Unlike ROE (H221, which is equity-relative and can be gamed by leverage), ROA reflects genuine operational improvement.

Signal: ROA_t − ROA_{t−1y} where ROA_t = NI_t / avg(TA_t, TA_{t−1y}). Sector-neutral top 20% by ROA improvement, monthly rebalance.

Expected regime behavior:
- **2000–2002 tech crash**: Improving ROA firms have genuine earnings improvement — not just multiple expansion.
- **2008–2009 GFC**: Balance-sheet stress hits ROA; firms maintaining positive ROA improvement are survivors.
- **2020 COVID**: Sharp ROA divergence between sectors; sector-neutral design is critical here.
- **2013–2017 low-vol**: Stable corporate earnings growth; ROA improvement should be systematically rewarded.

Success criteria: OOS Sharpe > 0.51, CI lower bound > 0. Expect H001 corr near zero (pure accounting signal).
Falsification: If ROA improvement is subsumed by ROE (H221) or net margin (H238) individually, this would show no incremental predictive power.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.751 |
| 95% CI | [0.339, 1.167] |
| CAGR | 8.66% |
| Max DD | −36.54% |
| H001 corr | −0.020 |
| Decision | **KEEP** |

**Sensitivity (window):** 126d → 0.749, 252d → 0.751, 504d → 0.712. Very flat profile for 126/252d; longer lookback (504d) shows slight decay.

**Folds 6–11 Sharpe:** 0.449, 1.816, 0.974, 1.806, 1.210, 0.997. Folds 0–5 zero (EDGAR coverage).

DuPont interpretation confirmed: ROA improvement (combining margin + turnover) delivers a robust signal in line with its components (H238 net margin: 0.696; H230 asset turnover: 0.741), with intermediate performance. The ROA composite doesn't dramatically outperform either component individually — suggesting the two effects are partially correlated in the cross-section.
