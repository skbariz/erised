# H180: EarningsTrendLong

## Hypothesis

**Signal:** YoY net income growth rate (SUE proxy). For each stock, compute `(NI_t - NI_{t-252}) / |NI_{t-252}|`. Rank cross-sectionally within each GICS sector; go long top 20% (highest YoY NI growth).

**Why it should work:** Post-earnings-announcement drift (PEAD) is one of the most replicated anomalies in academic finance (Ball & Brown 1968, Bernard & Thomas 1989). Stocks with high standardized unexpected earnings (SUE) continue to drift upward for 30–60 days post-announcement. Since we use EDGAR XBRL filing dates rather than actual press-release dates, this is a post-filing drift test: the signal fires 30–60 days after the press release, so it captures the late-filing-adjustment portion of PEAD rather than the initial gap. This is a weaker but still documented phenomenon.

**Sector-neutral:** S&P 500 sectors have very different earnings profiles. Tech/growth sectors typically show higher NI growth rates than Utilities/Staples. Sector-neutral ranking isolates idiosyncratic earnings quality from sector beta.

**Expected regime behavior:**
- **2000-2002 tech crash:** Growth sectors report deteriorating NI; strategy should rotate toward Financials/Energy/Staples that maintained earnings. Possible defensive benefit.
- **2008-2009 GFC:** Widespread NI deterioration across all sectors. Coverage drops significantly (EDGAR data sparse pre-2009); expect zero weights in folds 0-4.
- **2020 COVID:** Sharp NI drops in Travel/Retail; Healthcare/Tech maintained or grew. Strategy should favor the latter.
- **Low-vol 2013-2017:** Strong broad-market environment; NI growth signal should have positive but modest alpha over SPY.

**Data constraint:** EDGAR XBRL filing data starts ~2009. Folds 0-4 (test periods 1998-2009) will produce zero weights. This is expected and handled gracefully by the framework.

## Success criteria

- OOS Sharpe > 0.51 (beat SPY benchmark)
- 95% CI lower bound > 0
- MaxDD < -55.92% (better than SPY)
- H001 corr ideally < 0.70 (cross-sectional ≠ trend-following)

## Falsification

If the SUE proxy (YoY NI growth) produces Sharpe ≤ 0.51 or CI clips zero, this would indicate that: (a) EDGAR filing-date drift is too weak/late after the press-release announcement, (b) sector-neutral ranking within S&P 500 dilutes the signal, or (c) costs at 2.5 bps/turn eliminate the alpha given monthly rebalancing frequency.

## Implementation

- `EarningsTrendLong` in `src/erised/strategies/cross_sectional.py`
- Signal: `sue = (ni - ni.shift(252)) / ni.shift(252).abs()`
- Sector-neutral, within_sector_quantile=0.20, monthly rebalance
- Sensitivity sweep: yoy_window ∈ {126, 189, 252} days

## OOS Results

**Decision: KEEP**

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.715 |
| 95% CI | [0.311, 1.125] |
| CAGR | 8.31% |
| Max DD | -37.19% |
| H001 corr | +0.581 |

Folds 0-5 zero (pre-EDGAR + GFC-era coverage gap). Folds 6-11 all positive, ranging 0.370–2.034. Sensitivity flat across yoy_window values (126d/189d/252d all ~0.70). MaxDD -37.19% is meaningfully better than SPY (-55.92%).
