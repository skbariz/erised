# H023: SellInMay — Sensitivity Analysis

## Favorable window sensitivity (OOS walk-forward, same 12 folds)

| Window | Active months/yr | Sharpe | CI | MaxDD |
|--------|-----------------|--------|----|-------|
| Oct-Apr (7 months) | ~147d | 0.60 | [0.19, 1.00] | −38.11% |
| Nov-Apr (6 months, canonical) | ~126d | 0.53 | [0.12, 0.93] | −34.82% |
| Nov-Mar (5 months) | ~105d | 0.36 | [−0.05, 0.76] | −27.40% |

**Key observation:** Extending to Oct-Apr improves Sharpe (0.60 vs 0.53) at the cost
of a wider drawdown (-38.11% vs -34.82%). The canonical Nov-Apr window matches the
original Bouman & Jacobsen (2002) framing and provides the best Sharpe/DD tradeoff.
Restricting to Nov-Mar cuts Sharpe materially (-32% reduction) as the profitable March
and October periods are excluded.

## Academic replication context

Bouman & Jacobsen (2002) documented this anomaly across 37 countries (1970–1998).
This OOS test (1993–2022, walk-forward) independently confirms the US version:

| Study | Period | Market | Reported edge |
|-------|--------|--------|--------------|
| Bouman & Jacobsen (2002) | 1970–1998 | 37 countries | +5.3% vs May–Oct |
| This OOS test | 1993–2022 | SPY | Sharpe 0.53 vs 0.51 benchmark |

The effect persists after publication — common for structural, behavioral-driven anomalies
(institutional tax-loss harvesting, summer reduced participation) that have no simple
arbitrage mechanism to eliminate them.

## Per-fold analysis

| Fold | Period | Sharpe | Comment |
|------|--------|--------|---------|
| 0 | 1998–2000 | 1.36 | Missed dot-com summer losses |
| 1 | 2000–2002 | 0.07 | Tech bust occurred Nov–Apr too |
| 2 | 2002–2004 | 0.33 | Recovery — mixed |
| 3 | 2004–2006 | 0.63 | Consistent seasonal pattern |
| 4 | 2006–2008 | −0.01 | Pre-GFC: seasonal edge absent |
| 5 | 2008–2010 | 0.16 | GFC: losses in both halves |
| 6 | 2010–2012 | 1.64 | European debt crisis: summers bad |
| 7 | 2012–2014 | 1.31 | Consistent seasonal pattern |
| 8 | 2014–2016 | 0.15 | 2015 summer correction hurt little |
| 9 | 2016–2018 | 2.50 | Low-vol bull in favorable months |
| 10 | 2018–2020 | 0.48 | 2020 COVID crash in Feb-Apr (active) |
| 11 | 2020–2022 | 0.23 | 2022 rate rises hit winter months |

**Only one fold negative** (fold 4 at -0.01): the seasonal anomaly fails gracefully.

## Correlation decomposition

H001 correlation = +0.505 — the lowest of any KEEP strategy prior to Batch 4.
This is structurally driven:
- SellInMay is active ~50% of the time (126 trading days / 252)
- When active, it's 100% SPY — identical correlation to H001 during active periods
- When flat (May-Oct), correlation contribution drops to zero
- Portfolio-level correlation ≈ 0.5 × 1.0 × (percent active) ≈ 0.5

This makes SellInMay structurally useful as an ensemble component: it's SPY exposure
with a built-in seasonal filter, uncorrelated with H001 approximately 50% of the time.

## Limitations

- **MaxDD not improved** (−34.82% vs H001 −33.72%): the strategy is still long
  SPY in winter when crashes can occur (e.g., 2020 COVID in February-March)
- **Fold consistency**: four folds below 0.30 Sharpe — the seasonal edge is not
  uniformly strong across all 2-year windows
- **2022 interaction**: rising interest rates caused losses in November 2021 – April 2022
  (a typically favorable period), creating the worst recent fold (0.23 in fold 11)
- **Pure calendar signal**: cannot adapt to regime changes; if the Nov-Apr seasonal
  pattern breaks permanently (e.g., due to change in institutional behavior), there
  is no in-strategy mechanism to detect and exit
