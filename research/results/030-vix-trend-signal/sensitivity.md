# H030: VIXTrendSignal — Sensitivity Analysis

## MA period sensitivity (OOS walk-forward, same 12 folds)

| ma_period | N Trades | Sharpe | CI | MaxDD | H001 corr |
|-----------|---------|--------|----|-------|-----------|
| 10 | ~1,400 | −0.32 | [−0.73, 0.10] | −102.3% | −0.35 |
| 20 (canonical) | 940 | −0.09 | [−0.50, 0.31] | −83.24% | −0.28 |
| 60 | ~360 | −0.06 | [−0.46, 0.35] | −64.11% | −0.22 |

**Key observation:** Increasing the MA period reduces whipsaw (fewer trades, lower
MaxDD, Sharpe slightly less negative) but never reaches positive territory. The
conceptual signal is sound but transaction costs (940+ trades × 1.5 bps) and
the asymmetry of long vs short returns in a bull market prevent profitability.

## Why the first negative H001 correlation in the program

H030 is the only strategy in H001–H032 with a **negative correlation to H001** (−0.284).

This is structural:
- H001 is long when SPY trend is up (MA golden cross)
- H030 is short when VIX is rising
- VIX and SPY have ~−0.75 correlation
- Therefore: VIX rising → H030 short → H030 returns inversely correlated with SPY returns

The negative correlation makes H030 theoretically valuable as an ensemble hedge — but
only if the strategy itself had positive Sharpe. With Sharpe = −0.09, combining H030
with H001 would reduce returns without providing sufficient diversification benefit.

## Bull market long/short asymmetry

A long/short strategy assumes symmetric return opportunities. But SPY drifts upward
at ~8% CAGR. In a long/short strategy:

| Regime | H030 position | SPY return | H030 gross return |
|--------|--------------|-----------|------------------|
| VIX rising (bear) | Short | ~−15% | +15% |
| VIX falling (bull) | Long | ~+20% | +20% |

The problem: SPY bull periods last longer than bear periods. Even if individual
trades are equally profitable, the long periods dominate in total return — but
the strategy spends significant time short during bull market VIX retracements.

## Transaction cost analysis

| ma_period | Trades/yr | Cost at 1.5bps | Net drag |
|-----------|-----------|----------------|----------|
| 10 | ~116 | −1.74% | Dominates any gross edge |
| 20 | ~78 | −1.17% | Still eliminates most alpha |
| 60 | ~30 | −0.45% | Better, but Sharpe still negative |

At ma_period=60: 360 trades / 12 years ≈ 30 trades/yr ≈ 0.45% annual drag.
The underlying gross Sharpe (before costs) would need to be ~0.5 to break even
after costs — which historical VIX trend signals don't reliably achieve on SPY.

## Long-only variant comparison

If H030 went flat (0) instead of short (−1) when VIX rising:
- This would be equivalent to VIXRegimeFilter with MA smoothing
- Estimated Sharpe ≈ 0.40–0.45 (similar to H029: 0.43)
- The shorting component actively destroys value in this configuration

## Per-fold analysis

| Fold | Period | Sharpe | Comment |
|------|--------|--------|---------|
| 0 | 1998–2000 | 0.30 | Dot-com: VIX trend sometimes aligned |
| 1 | 2000–2002 | −0.08 | Short during partial recoveries |
| 2 | 2002–2004 | −0.24 | Recovery period: strategy short |
| 3 | 2004–2006 | 0.14 | Low-vol: mostly long |
| 4 | 2006–2008 | −0.12 | Pre-GFC: shorting at wrong times |
| 5 | 2008–2010 | −0.15 | GFC: shorting during market recovery |
| 6 | 2010–2012 | 0.14 | Mixed |
| 7 | 2012–2014 | 0.21 | Bull market: VIX declining → long |
| 8 | 2014–2016 | 0.04 | Near-flat |
| 9 | 2016–2018 | −0.17 | Low-vol bull: VIX noise → many losses |
| 10 | 2018–2020 | −0.27 | 2018-2020 volatility: net negative |
| 11 | 2020–2022 | −0.15 | COVID + 2022: mixed whipsaw |

**Eight negative or near-zero folds** — the signal is unreliable across all regimes.

## Unique value: negative H001 correlation

Despite being killed, H030 identifies something important: a negative-correlation
strategy concept exists. In future batches (H038: Hurst exponent, H039: autocorrelation),
similar negative-correlation mechanisms may appear with better Sharpe. The H030
result proves that VIX-trend shorting captures a genuinely different return source —
the implementation challenge is making it profitable after costs.

## Limitations

- **Transaction costs are fatal**: 940 trades at 1.5 bps = ~1.4% annual drag
- **Bull market asymmetry**: short SPY during VIX retracements in bull markets loses frequently
- **Mean-reverting VIX signal**: VIX trends are shorter and choppier than equity price trends, causing whipsaw at high frequency
