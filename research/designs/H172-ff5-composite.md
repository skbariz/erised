# H172: FF5Composite

## Hypothesis
(1/3)*z(B/M) + (1/3)*z(GP/BE) + (1/3)*z(-ΔTA/TA), sector-neutral.
Combines the three non-market Fama-French 5-factor signals:
- Value: HML (book-to-market)
- Profitability: RMW (gross profit / book equity)
- Investment: CMA (low asset growth, negated so high value = low growth)

Diversification benefit: the three signals are partially orthogonal.
Cheap stocks aren't always profitable; profitable stocks don't always invest conservatively.
Combining them should reduce noise.

## Expected regime behavior
- Should benefit from each component in different market regimes
- Value: outperforms in recovery/mean-reversion
- Profitability: outperforms in growth regimes
- Investment: outperforms when capex discipline is rewarded

## Success criteria
- OOS Sharpe > 0.70 (composite should exceed individual components)
- H001 corr < 0.75

## What would falsify this
- Sharpe below single-factor strategies (no diversification benefit)
- Value drag overwhelms quality/investment signals in post-2010 growth market
