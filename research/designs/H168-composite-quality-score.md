# H168: CompositeQualityScore (QMJ-Inspired)

## Hypothesis
Multi-dimensional quality composite combining three orthogonal quality dimensions:
1. Profitability: avg z(GP/A, ROE, cash-ROE) — current earnings power
2. Growth: z(1-year change in GP/A) — profitability trend
3. Safety: avg z(-beta, -LTD/TA, current_ratio) — balance sheet / market risk

The safety dimension uses rolling 252d beta (from prices) and balance sheet ratios (from fundamentals).
Key insight: safety dimension is always computable (prices available from 1993),
so the strategy should produce non-zero weights even in pre-XBRL folds.

## Expected regime behavior
- 2000-2002 tech crash: low-beta, profitable, cash-rich firms dramatically outperform
- 2008-2009 GFC: safety dimension (low leverage, high current ratio) should protect
- 2020 COVID: mixed — safety screen reduces airline/hotel exposure but may miss tech rebound
- 2013-2017: profitable growing firms with manageable beta should perform well

## Success criteria
- OOS Sharpe > 0.80 (expect higher than single-signal strategies)
- Non-zero returns in pre-EDGAR folds (safety dimension carries early period)
- CI lower bound > 0.30 (strong signal)

## What would falsify this
- Pre-EDGAR folds still zero (beta/leverage signal not contributing)
- Composite no better than individual dimensions (diversification benefit absent)

## Results (2026-05-25)
- Sharpe: 1.181  CI: [0.773, 1.594]  CAGR: 18.55%  MaxDD: -40.44%
- H001 corr: +0.6596
- Decision: **KEEP**
- Notes: OUTSTANDING. Works in ALL 12 folds (1.258, 2.242, 1.477, 2.169, ...).
  The safety dimension (beta from prices) provides signal even in pre-EDGAR period.
  Sharpe 1.181 exceeds the H150-H199 individual strategy program best.
  Sensitivity monotone: tighter quantile → higher Sharpe (10%: 1.282, 20%: 1.181, 30%: 1.089).
