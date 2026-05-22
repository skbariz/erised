# H016 — Volatility-Scaled Trend: Stats

**Status:** kept
**Family:** risk management / trend
**Instrument:** SPY
**Run date:** 2026-05-21

## Parameters

| Parameter | Value |
|-----------|-------|
| Fast MA | 50 days |
| Slow MA | 200 days |
| Target vol | 10% annualized |
| Vol lookback | 20 days |
| Max leverage | 1.0× (no leverage) |
| Sizing | continuous [−1, +1] |

## Walk-forward configuration

| Setting | Value |
|---------|-------|
| Research period | 1993-01-29 to 2022-12-30 |
| Train window | 5 years |
| Test window | 2 years |
| Folds | 12 |
| Commission | 0.5 bps |
| Slippage | 1.0 bps |

## OOS metrics (combined across 12 folds)

| Metric | H016 | SPY benchmark | Delta |
|--------|------|---------------|-------|
| Sharpe | 0.55 | 0.51 | +0.04 |
| 95% CI on Sharpe | [0.14, 0.95] | — | — |
| CAGR | 5.33% | 8.35% | −3.02 pp |
| Max drawdown | −26.30% | −55.92% | −29.62 pp |
| Trades (position changes) | 4,522 | — | — |

## Sensitivity (target_vol)

| target_vol | Sharpe |
|------------|--------|
| 8% | 0.55 |
| **10%** | **0.55** |
| 15% | 0.55 |

Sharpe is flat across vol targets because the signal (trend direction) does not change; only position magnitude changes. Higher targets produce higher CAGR but also higher drawdowns in proportion.

## Per-fold OOS Sharpe

`['0.19', '0.70', '1.10', '0.38', '0.81', '0.86', '-0.24', '1.29', '-0.16', '1.34', '0.19', '0.41']`

## Correlation with H001: +0.788

## Decision: kept

**Best max drawdown in the entire research program: −26.30%** (vs H001 −33.72%, H008 −31.23%, SPY −55.92%).

Three keep criteria met:
1. OOS Sharpe 0.55 > SPY 0.51 ✓
2. 95% CI [0.14, 0.95] — lower bound positive ✓
3. Max DD −26.30% < SPY −55.92% ✓

The vol-targeting overlay works as intended: in high-volatility regimes (crashes, crises), realized vol rises well above the 10% target, scaling the position toward zero before the worst losses accumulate. In calm markets, full (or near-full) exposure is maintained.

Sharpe (0.55) is lower than H001 (0.61) because CAGR is reduced (5.33% vs 7.86%) by the downsizing in volatile periods. But max DD improves dramatically — this is the trade-off of vol targeting.

**Ensemble note:** Correlation +0.788 with H001 is high — H016 is capturing the same underlying trend signal as H001, just with better position sizing. It does not add ensemble diversification but does improve the max-DD profile significantly. Worth keeping as a standalone risk-managed alternative to H001.
