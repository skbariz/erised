# H295: ExecutionEfficientComposite — GP/TA + Investment Efficiency, Bi-Monthly

## Hypothesis

The most durable cross-sectional signals are gross profitability (H227) and investment
efficiency (H287). Combining them with execution-aware rebalancing (bi-monthly + buy/hold
band) should maximize NET Sharpe after cost model. The 2-factor combination avoids the
dilution risk of 3+ factor composites.

Signal: score = z(gp_ta) + z(efficiency_ratio)
where gp_ta = GP/TA and efficiency_ratio = rev_growth / (capex_intensity + 0.01).
Rebalance every 42 business days (bi-monthly). Buy when rank ≥ 0.70 (top-30%).
Hold until rank falls below 0.50 (sell only below top-50%). Sector-neutral.

Distinction from H290 (3-factor capital quality composite): H290 adds NOA as a third factor
at the dilution boundary. H295 uses only 2 factors and focuses on execution efficiency
(bi-monthly cadence + buy/hold band) as the primary value-add over standalone signals.

Expected regime behavior:
- **2000–2002**: Quality + efficiency filters remove capital-destroying tech names.
- **2008–2009**: GP and revenue both stressed; efficiency ratio disrupted.
- **2013–2017**: Stable investment cycle + strong GP. Best environment.
- **2020**: Revenue shock; efficiency ratio disrupted temporarily.

Success criteria: OOS Sharpe ≥ 0.92, CI lower > 0, beats H227 (0.878) and H287 standalone.
Falsification: Bi-monthly cadence hurts vs monthly — quality signals also decay within 42 days.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
