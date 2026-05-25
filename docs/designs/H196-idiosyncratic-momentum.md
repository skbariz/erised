# H196: IdiosyncraticMomentum

## Hypothesis
Standard 12-1m momentum conflates stock-specific skill with sector-level trends. A stock in a hot sector will appear to have momentum even if it's underperforming within that sector. Idiosyncratic momentum = stock_12m_return - sector_equal_weight_12m_return strips the systematic sector component and isolates stock-level momentum. This should be less correlated with H001 and more predictive of relative performance within sectors. Sector-neutral top 20%, monthly rebalance.

## Expected regime behavior
- **2000-2002 tech crash**: Sector-adjusted signal avoids lumping all tech together; relative winners within tech may still outperform
- **2008-2009 GFC**: Similar protection via sector isolation during uniform sell-offs
- **2013-2017 low-vol**: Stock-specific momentum should persist in calm trending regimes
- **2020 COVID**: Rapid sector rotations may confuse the signal; 21-day skip provides some stability

## Success criteria
- OOS Sharpe > 0.51
- CI lower bound > 0
- H001 corr < 0.75 (sector adjustment should reduce trend correlation)

## Falsification
- If performance is indistinguishable from H150 (raw momentum), the sector adjustment adds no value
- If sector-average subtraction introduces too much noise, OOS variance will dominate

## Parameters
- `mom_window`: 252 days
- `skip_window`: 21 days
- `within_sector_quantile`: 0.20
- `rebalance_period`: 21 days
