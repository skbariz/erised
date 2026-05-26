# H238: NetProfitMargin — YoY Improvement in NI/Revenue

## Hypothesis

Improving net profit margin (NI/revenue) captures bottom-line profitability efficiency trends. Unlike gross margin (which captures pricing power) or ROE (which is equity-relative), net profit margin reflects the full operating and financial cost structure. A rising NI/revenue signals either cost discipline, pricing improvements, or reduced financing costs — all positive quality signals.

Expected regime behavior:
- **2000–2002 tech crash**: Tech firms with negative margins get punished; improving-margin firms survive.
- **2008–2009 GFC**: Cyclical margin compression; firms already improving margins pre-crisis do better.
- **2020 COVID**: Margin bifurcation — some sectors benefit (tech, healthcare), others crushed.
- **2013–2017 low-vol**: Stable earnings growth period; margin improvement should be rewarded.

Success criteria: OOS Sharpe > 0.51, CI lower bound > 0, H001 corr < 0.10.
Falsification: If the net margin signal adds nothing beyond GP/revenue (H239), expect Sharpe ≈ H239 but with higher noise from below-the-line items (interest, taxes).

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.696 |
| 95% CI | [0.293, 1.095] |
| CAGR | 8.00% |
| Max DD | −37.21% |
| H001 corr | −0.020 |
| Decision | **KEEP** |

**Sensitivity (window):** 126d → 0.675, 252d → 0.696, 504d → 0.667. Modest window sensitivity; 252d canonical.

**Folds 6–11 Sharpe:** 0.424, 1.869, 0.809, 1.573, 1.025, 0.983. Folds 0–5 zero (EDGAR ~2009+).

Lowest Sharpe in Batch 8 (0.696), consistent with hypothesis that below-the-line noise (taxes, interest) dilutes the signal relative to gross margin level (H239: 0.861). Passes the keep bar cleanly with CI lower bound 0.293 > 0.
