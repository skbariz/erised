# H281: QuarterlyEarningsDrift — NI Improvement as PEAD Proxy

## Hypothesis

DellaVigna & Pollet (2009) show that earnings announcements during low-attention periods
generate a drift window that persists longer. Without earnings announcement dates, we detect
quarterly filings as jumps in the forward-filled fundamental data. Stocks with positive
NI improvement (NI_t > NI_{t-4}) are experiencing earnings "surprises" relative to the
prior year's same quarter. We test whether this improvement signal produces a sustained drift.

Signal: Compute annual earnings change = (NI - NI.shift(252)) / abs(NI.shift(252)).clip(-5, 5).
Rank descending. Long top-quintile (largest NI improvement). Sector-neutral, monthly rebalance.

This is a YoY quarterly earnings improvement signal using the forward-filled fundamental data.
The quarterly jump date serves as an implicit announcement proxy.

Expected regime behavior:
- **2000–2002**: Negative NI surprises dominate; strategy defensively underweighted.
- **2008–2009**: Broad NI deterioration; few positive surprise stocks.
- **2013–2017**: Steady earnings growth; signal consistently positive.
- **2020**: COVID shock collapses NI; recovery in H2 2020 generates strong signal.

Success criteria: OOS Sharpe ≥ 0.70, CI lower > 0. Check H239 (NI level) corr < 0.85.
Falsification: H239 corr > 0.85 — NI change is just a lagged version of NI level in S&P 500.

## Results

| Metric | Value |
|--------|-------|
| OOS Sharpe | |
| 95% CI | |
| CAGR | |
| Max DD | |
| H001 corr | |
| Decision | |
