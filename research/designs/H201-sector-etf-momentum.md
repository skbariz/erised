# H201: SectorETFMomentum

**Signal:** Moskowitz & Grinblatt (1999) — industry momentum applied to sector/style ETFs.
**Date:** 2026-05-25

## Hypothesis

Sector ETFs carry persistent 6-month momentum: top-performing sectors continue to outperform for the next 1-3 months. Rotating into the top-3 equity ETFs (XLP, XLU, IVE, IVW, QQQ, IWM, VNQ) by 6m trailing return and risk-parity weighting them should outperform SPY with lower drawdown, because the strategy exits sectors experiencing fundamental deterioration.

## Expected behavior by regime

- 2000-2002 tech crash: exits IVW/QQQ into defensive XLP/XLU or value IVE — strong signal
- 2008 GFC: exits all equity ETFs eventually; VNQ worst; should rotate toward defensive
- 2020 COVID: sharp sector rotation (tech/QQQ rallies fast); captures recovery
- 2013-2017 low vol: stable rotation, low turnover, likely QQQ/IVW dominant

## Success criteria

OOS Sharpe > 0.60, CI lower > 0, MaxDD better than SPY (-55.92%).
H001 corr < 0.70 (sector rotation should differ from simple equity trend-following).

## What would falsify this

If H001 corr > 0.85, sector ETF rotation collapses to equity long/short. If Sharpe < H021, the ETF universe is too small (7 assets) to offer meaningful diversification vs risk-parity.

## Results

**Decision: KEEP**

| Metric | Value |
|--------|-------|
| OOS Sharpe | 0.654 |
| 95% CI | [0.273, 1.057] |
| CAGR | 7.24% |
| Max DD | −38.99% |
| H001 corr | +0.068 |

Sensitivity (top_n): top_n=1 Sharpe 0.606, top_n=2 Sharpe 0.612, top_n=3 Sharpe 0.654, top_n=4 Sharpe 0.626, top_n=5 Sharpe 0.606.
Sensitivity (mom_window): 63d Sharpe 0.595, 126d Sharpe 0.654, 189d Sharpe 0.667, 252d Sharpe 0.636. Results stable across windows.

All success criteria met. CI lower bound > 0. MaxDD (−38.99%) well above SPY (−55.92%). H001 corr +0.068 — genuine independence from equity trend-following. AA strategy suitable for cross-program ensemble inclusion.
