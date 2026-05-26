# H247: RetainedEarningsProxy — YoY Book Equity Growth / Total Assets

## Hypothesis

Book equity grows when a company is profitable AND retaining those profits rather than distributing them. The ratio (BE_t − BE_{t-1y}) / TA_{t-1y} captures both: it requires positive earnings (otherwise BE falls) and capital discipline (paying out all earnings would leave BE flat). This is a "quality earnings" proxy that avoids pure accounting metrics.

Signal: (BE − BE.shift(252)) / TA.shift(252), clipped [-0.5, 1.0]. Sector-neutral top 20%, monthly rebalance.

Expected regime behavior:
- **2000–2002 tech crash**: Profitable companies retaining earnings are real businesses; avoids pre-revenue firms.
- **2008–2009 GFC**: Earnings collapse hits retained earnings growth; relative ranking identifies resilient firms.
- **2020 COVID**: BE growth bifurcates by sector; sector-neutral design essential.
- **2013–2017 low-vol**: Steady quality premium expected.

Success criteria: OOS Sharpe > 0.51, CI lower bound > 0. Distinct from shareholder yield (H207, buybacks) and asset growth (H206, total asset expansion).

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.783 |
| 95% CI | [0.376, 1.191] |
| CAGR | 9.28% |
| Max DD | −37.85% |
| H001 corr | −0.017 |
| Decision | **KEEP** |

**Sensitivity (window):** 126d → 0.751 | 252d → 0.783 | 504d → 0.730. Moderate sensitivity; 252d canonical is the peak.

**Folds 6–11 Sharpe:** 0.662, 1.998, 0.983, 1.544, 1.114, 1.075. Folds 0–5 zero.

Solid performance with consistent fold Sharpe above 0.66 in all active folds. The 252d window outperforms both shorter and longer lookbacks, suggesting the 1-year retained earnings signal is well-calibrated.
