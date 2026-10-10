# H181: RevenueEarningsDoubleGrowth

## Hypothesis

**Signal:** Jegadeesh-Livnat (2006) proxy. Require both net income AND revenue to have grown YoY. Composite signal = `(z_score(NI_growth) + z_score(Rev_growth)) / 2` where both growth rates are positive. Rank within sector; long top 20%.

**Why it should work:** Jegadeesh and Livnat (2006) show that revenue surprises have independent predictive power beyond earnings surprises. The dual-filter creates a quality screen: a company reporting rising earnings on rising revenue is a fundamentally stronger business than one showing NI growth via cost-cutting alone (rising NI, falling revenue). This composite should have better signal quality (lower noise) at the cost of fewer eligible stocks.

**Sector-neutral:** Same reasoning as H180. Composite z-score is computed cross-sectionally within sectors, then stocks must clear the "both positive" gate.

**Expected regime behavior:**
- **2000-2002 tech crash:** Tech bubble involved revenue growth without NI growth (many companies). The double-positive filter may actually help here by excluding loss-making tech.
- **2008-2009 GFC:** Both NI and revenue collapsed broadly. Same data-gap issue as H180 for early folds.
- **2020 COVID:** Sharp bifurcation: tech/healthcare grew both metrics; travel/energy fell. Good signal.
- **Low-vol 2013-2017:** Consistent broad earnings growth; signal should be positive but moderate.

**Data constraint:** Same as H180. Folds 0-4 produce zero weights. Additional constraint: requires both net_income AND revenue data for a stock to enter the universe.

## Success criteria

- OOS Sharpe > 0.51
- 95% CI lower bound > 0
- Expect higher Sharpe than H180 (cleaner signal) with similar or lower trade count
- H001 corr < 0.70

## Falsification

If Sharpe ≤ H180 Sharpe, the dual-positive filter adds no value over pure NI growth — revenue growth is redundant conditional on NI growth within S&P 500. Kill.

## Implementation

- `RevenueEarningsDoubleGrowth` in `src/erised/strategies/cross_sectional.py`
- Signal: `composite = (z(ni_growth) + z(rev_growth)) / 2` where `both_growth > 0`
- Sensitivity sweep: within_sector_quantile ∈ {0.10, 0.20, 0.30}

## OOS Results

**Decision: KEEP**

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.762 |
| 95% CI | [0.355, 1.172] |
| CAGR | 8.70% |
| Max DD | -36.80% |
| H001 corr | +0.586 |

Beats H180 (0.762 vs 0.715) — Jegadeesh-Livnat dual-filter validated. Folds 0-5 zero (pre-EDGAR). Folds 6-11 all positive (0.535–2.119). Sensitivity flat across quantile thresholds. MaxDD -36.80% slightly better than H180.
