# H250: GrandFundamentalComposite — Terminal 5-Factor Composite

## Hypothesis

The terminal hypothesis for the cross-sectional fundamental program: combine the five best-performing families in equal z-score weights: Novy-Marx gross profitability (GP/TA), Sloan accruals quality (-delta(NWC)/avg_TA), conservative investment (-asset_growth), cash generation (OCF/TA), and low leverage (-LTD/TA).

Each factor has individually passed the keep bar in prior batches. Together they should form a diversified fundamental quality composite with more stable fold-to-fold performance and potentially higher Sharpe.

Signal: z(GP/TA) + z(-accruals) + z(-asset_growth) + z(OCF/TA) + z(-LTD/TA). Sector-neutral top 20%, monthly rebalance.

Expected regime behavior:
- **2000–2002 tech crash**: GP/TA and accruals signal should be activated.
- **2008–2009 GFC**: Low leverage and OCF signals most relevant.
- **2020 COVID**: Broad quality screen; GP/TA and OCF should dominate.
- **2013–2017 low-vol**: All 5 components contribute.

Success criteria: OOS Sharpe > 0.51, CI lower bound > 0. Expect 0.85–0.95 if diversification works; may underperform best individual signals if they partially conflict.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.742 |
| 95% CI | [0.329, 1.144] |
| CAGR | 8.61% |
| Max DD | −35.08% |
| H001 corr | −0.020 |
| Decision | **KEEP** |

**Sensitivity (within_sector_quantile):** 10% → 0.699 | 20% → 0.742 | 30% → 0.740. Unusual: top 10% is the weakest; concentration hurts this composite.

**Folds 6–11 Sharpe:** 0.058, 2.211, 0.803, 2.206, 1.135, 1.008. Fold 6 (2010–2012) is extremely weak (0.058), dragging the overall Sharpe down significantly.

The terminal composite (0.742) underperforms the single-factor champion H239 GrossMarginLevel (0.861) and the 2-factor composites in Batch 7. Fold 6 weakness likely reflects early EDGAR data noise when multiple signals are combined: if GP/TA data starts at different times than accruals data, the composite gets confused in the transition period. Nonetheless, it passes the keep bar and achieves the best MaxDD (−35.08%) in Batch 10 at 30% quantile. The lesson: more factors ≠ better; the strongest single signal (GP/TA) can outperform over-diversified composites.
