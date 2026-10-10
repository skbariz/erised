# H244: AltmanZProxy — Simplified Altman Z-Score from Available Fundamentals

## Hypothesis

The Altman Z-score (1968) is the canonical bankruptcy prediction model. Our proxy uses only available EDGAR fundamentals: NWC/TA (liquidity), NI/TA (profitability), BE/(TA-BE) (equity-to-liabilities leverage), and revenue/TA (asset efficiency). High Z-proxy companies have lower distress risk and should outperform.

Signal: z(NWC/TA) + z(NI/TA) + z(BE/TL) + z(rev/TA). Sector-neutral top 20%, monthly rebalance.

Expected regime behavior:
- **2000–2002 tech crash**: Pre-crash tech companies score poorly on NI/TA and BE/TL; signal avoids them.
- **2008–2009 GFC**: Highly leveraged financials score poorly; financial sector gets downweighted.
- **2020 COVID**: Diversified distress signal; provides broad quality screening.
- **2013–2017 low-vol**: Quality premium in stable environments.

Success criteria: OOS Sharpe > 0.51, CI lower bound > 0. Expect similar Sharpe to single-factor solvency signals since Z-score aggregates them.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.764 |
| 95% CI | [0.362, 1.168] |
| CAGR | 9.48% |
| Max DD | −36.64% |
| H001 corr | −0.017 |
| Decision | **KEEP** |

**Sensitivity (within_sector_quantile):** 10% → 0.779 | 20% → 0.764 | 30% → 0.781. Flat sensitivity; composite score is concentration-insensitive.

**Folds 5–11 Sharpe:** 0.221, 0.618, 2.252, 0.793, 1.719, 0.948, 1.095. Fold 7 Sharpe 2.252 is the highest in Batch 9.

4-component composite delivers results in line with individual solvency factors, with fold 7 Sharpe being exceptional. The equal-weighted z-score approach is well-calibrated — no single component dominates.
