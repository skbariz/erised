# H031: RealizedVsImpliedVol — Sensitivity Analysis ★

## Realized window sensitivity (OOS walk-forward, same 12 folds)

| realized_window | Active fraction | Sharpe | CI | MaxDD |
|-----------------|----------------|--------|----|-------|
| 10d | ~68% | 0.61 | [0.20, 1.01] | −49.82% |
| 20d (canonical) | ~67% | 0.58 | [0.16, 0.96] | −51.43% |
| 60d | ~64% | 0.42 | [0.01, 0.83] | −44.12% |

**Key observation:** Shorter realized windows (10d) slightly outperform (Sharpe 0.61
vs 0.58 canonical) because they respond faster to volatility regime changes. The 20d
window is a conventional choice balancing responsiveness and noise. Longer windows
(60d) underperform: when implied vol remains elevated for weeks after a crisis,
the 60d realized vol eventually exceeds VIX and forces the strategy flat, missing
the early recovery.

## The Volatility Risk Premium (VRP) mechanism

The VRP = VIX − realized vol is historically positive on average:

| Period | VRP average | H031 active? |
|--------|------------|-------------|
| Low-vol regimes (VIX 12-18) | +5 to +10 pts | Yes (~90% of days) |
| Elevated-vol regimes (VIX 20-30) | +0 to +5 pts | Yes (~60% of days) |
| Crisis regimes (VIX > 30) | −5 to +20 pts | Mixed (depends on realization) |
| Post-crisis rebound | VRP turns positive fast | Yes (early signal) |

The VRP is positive ~67% of days in the research period — explaining the 67%
active fraction. The negative-VRP periods (realized > implied) coincide with
realized market turmoil that options had not fully priced in.

## Why H031 achieves CAGR 8.52% (highest in Batch 5)

The strategy is long SPY during ~67% of days. The active periods are positively
selected — the VRP filter tends to:
- Be active during low-vol bull markets (full equity participation)
- Turn flat just before or during realized vol spikes (early crash protection)
- Re-enter as realized vol starts falling from peaks (early recovery capture)

CAGR 8.52% vs SPY 8.35% benchmark: marginal CAGR improvement because the VRP
filter is right more than wrong about regime selection.

## MaxDD −51.43%: not a protection strategy

The MaxDD of −51.43% is close to SPY's −55.92%. The VRP filter provides some
crash protection but not consistently:
- 2008 GFC: VIX rose extremely fast; realized vol caught up, VRP turned negative
  only after significant losses. H031 was partially long during early GFC.
- 2020 COVID: VIX spiked rapidly; the 20d realized window hadn't caught up, so
  VRP was briefly very positive (VIX far > realized) — H031 was long when it should
  have been flat, then the VRP went negative and forced flat during recovery.

The filter has timing limitations in fast-crash environments where realized vol
lags implied vol by the entire lookback window.

## H001 correlation decomposition (+0.693)

| Component | Contribution to H001 correlation |
|-----------|----------------------------------|
| Both long SPY ~67% of days | High structural correlation |
| VRP filter prevents flash-crash exits | Less correlated in volatile periods |
| Both signals: trend-agnostic long-only | Forces correlation closer to 1 |

Correlation 0.693 means H031 provides limited diversification vs H001. The VRP
signal is a different *information source* (options market) but achieves a similar
*portfolio behavior* (long SPY ~2/3 of the time).

## Academic backing

Carr & Wu (2009) documented the variance risk premium across equity indices. The
VRP averages +4-6 vol points (VIX > realized) in normal markets, creating a
systematic positive-expected-return regime for equity risk-takers. This OOS test
confirms the regime-selection aspect holds on SPY daily data.

| Study | Finding |
|-------|---------|
| Carr & Wu (2009) | VRP positive on average; negative during crashes |
| Bollerslev et al. (2009) | VRP predicts equity returns at 1-month horizon |
| This OOS test (1993–2022) | Sharpe 0.58 in VRP-positive periods vs 0.51 benchmark |

## Per-fold analysis

| Fold | Period | Sharpe | Comment |
|------|--------|--------|---------|
| 0 | 1998–2000 | 0.63 | Low-vol era: mostly active |
| 1 | 2000–2002 | −0.30 | Tech bust: VRP positive as VIX rose |
| 2 | 2002–2004 | 0.46 | Recovery: mixed VRP regime |
| 3 | 2004–2006 | 0.58 | Low-vol: consistently active |
| 4 | 2006–2008 | 0.67 | Pre-GFC calm |
| 5 | 2008–2010 | 0.38 | GFC: VRP turned negative, partially flat |
| 6 | 2010–2012 | 0.83 | Post-GFC recovery: strong |
| 7 | 2012–2014 | 0.82 | Consistent bull + low vol |
| 8 | 2014–2016 | 0.48 | Moderate: 2015 correction |
| 9 | 2016–2018 | 0.83 | Low-vol bull: maximum active |
| 10 | 2018–2020 | 0.72 | Mixed: 2018 correction + recovery |
| 11 | 2020–2022 | 0.40 | COVID + 2022 rate rises |

**Only one negative fold** (fold 1: −0.30, tech bust). The strategy is positive in
11 of 12 folds — more consistent than any other Batch 5 strategy.

## Limitations

- **MaxDD −51.43%**: near-equivalent to SPY; no crash protection in fast selloffs
- **H001 correlation +0.693**: provides limited ensemble diversification
- **Timing lag**: 20d realized window means the VRP signal is 20 days behind actual volatility regimes — optimal for slow volatility changes, poor for flash crashes
