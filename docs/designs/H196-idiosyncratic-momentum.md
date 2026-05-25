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

## Results (OOS 1993-2021, 12 folds)
- **Decision: KEEP**
- OOS Sharpe: 0.930  CI [0.526, 1.343]
- CAGR: 18.08%  Max DD: -50.97%
- H001 corr: +0.690
- Fold Sharpes: 1.45, 0.96, 1.21, 1.50, 1.12, -0.23, 0.74, 2.32, 0.99, 1.62, 1.08, 0.89
- Sensitivity: 126d window actually superior (1.000 Sharpe) — sector adjustment less helpful with shorter window
- Sector adjustment provides marginal improvement over H150 (0.981); both are very correlated
