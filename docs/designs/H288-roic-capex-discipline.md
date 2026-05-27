# H288: ROICCapexDiscipline — ROIC Quality + Low Capex Growth Composite

## Hypothesis

The best capital allocators earn high returns on invested capital (ROIC) AND restrain further
investment below their returns. Combining ROIC quality with low capex growth creates a
capital-discipline composite: firms that earn well on existing assets and don't over-expand.

Signal: ROIC proxy = NI / (TA - CL).where((TA - CL) > 0) (net income / net assets).
capex_growth from H286 signal (low = better).
score = z(ROIC) + z(-capex_growth) = reward high ROIC, penalize high capex growth.
Long top-quintile. Sector-neutral, monthly rebalance.

Distinction from H287 (efficiency ratio): H287 uses revenue growth per capex. H288 uses
earnings quality (ROIC) combined with investment restraint. H287 rewards productivity;
H288 rewards discipline.

Expected regime behavior:
- **2000–2002**: High ROIC + low capex = disciplined survivors. Strong signal.
- **2008–2009**: ROIC collapses broadly; still informative within-sector.
- **2013–2017**: High ROIC + stable capex = steady outperformance.
- **2020**: ROIC disruption in Q1-Q2; discipline signal somewhat delayed.

Success criteria: OOS Sharpe ≥ 0.82, CI lower > 0, beats both H286 and standalone ROIC.
Falsification: Sharpe below both standalone components — ROIC and capex are negatively
correlated, destroying composite value.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
