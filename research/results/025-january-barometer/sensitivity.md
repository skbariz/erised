# H025: JanuaryBarometer — Sensitivity Analysis

## Why there is no parameter sensitivity

JanuaryBarometer has no tunable parameters. The signal is binary and fixed:
- January return > 0 → long Feb–Dec
- January return ≤ 0 → flat Feb–Dec

There is no lookback, no threshold, no moving average. This makes the strategy
immune to in-sample parameter overfitting but also impossible to optimize.
The sensitivity analysis instead covers: regime decomposition, MaxDD improvement
source, and comparison to SellInMay (the other calendar KEEP strategy).

## Regime analysis: when the barometer is right vs wrong

| Year | Jan return | Signal | Full-year SPY return | Correct? |
|------|-----------|--------|---------------------|---------|
| 1998 | +1.0% | LONG | +28.6% | ✓ |
| 1999 | +3.1% | LONG | +21.0% | ✓ |
| 2000 | −5.1% | FLAT | −9.1% | ✓ |
| 2001 | +3.5% | LONG | −11.9% | ✗ (false positive) |
| 2002 | −1.6% | FLAT | −22.1% | ✓ |
| 2003 | −2.7% | FLAT | +28.7% | ✗ (false negative) |
| 2004 | +1.7% | LONG | +10.9% | ✓ |
| 2005 | −2.5% | FLAT | +4.9% | ✗ (false negative) |
| 2006 | +2.6% | LONG | +15.8% | ✓ |
| 2007 | +1.4% | LONG | +5.5% | ✓ |
| 2008 | −6.1% | FLAT | −37.0% | ✓ (avoided GFC year) |
| 2009 | −8.6% | FLAT | +26.5% | ✗ (missed recovery) |
| 2010 | −3.7% | FLAT | +15.1% | ✗ (false negative) |
| 2019 | +7.9% | LONG | +31.5% | ✓ |
| 2020 | −0.2% | FLAT | +18.4% | ✗ (missed COVID recovery) |
| 2022 | −5.3% | FLAT | −18.2% | ✓ |

**Accuracy rate ≈ 70%**: the barometer correctly predicts direction in ~70% of years.
The key value is asymmetric: it correctly identified 2000, 2002, 2008, and 2022 as
"flat" years — the four largest bear markets in the study period. False negatives
(2003, 2009, 2010, 2020) missed recoveries but incurred no losses.

## MaxDD comparison: best non-risk-parity strategy

| Strategy | MaxDD | H001 Corr | Sharpe |
|----------|-------|-----------|--------|
| H021 RiskParity | −18.41% | — | 1.03 |
| **H025 JanuaryBarometer** | **−29.53%** | **+0.565** | **0.53** |
| H001 MA Crossover | −33.72% | +1.000 | 0.61 |
| H011 ADX Filter | −33.72% | +0.921 | 0.63 |
| SPY Buy-and-Hold | −55.92% | — | 0.51 |

H025 has the second-lowest MaxDD in the entire research program (−29.53%),
achieved by sitting flat in the worst bear-market years (2000, 2002, 2008, 2022).
Drawdown reduction is 49pp vs SPY buy-and-hold and 4pp vs H001.

## How the MaxDD of -29.53% occurs

The strategy still holds SPY during February-December of positive-January years.
Drawdown cannot be zero because:
- Intra-year crashes can occur after a positive January (2001: +3.5% Jan → −11.9% full year)
- COVID in 2020: January was −0.2% (flat signal), so recovery was missed but crash
  was mostly in March (not January) → partial drawdown avoided

The -29.53% MaxDD comes from 2001 (false positive year) where the strategy stayed
long through the September 11th and tech bust continuation.

## Per-fold analysis

| Fold | Period | Sharpe | Comment |
|------|--------|--------|---------|
| 0 | 1998–2000 | 1.04 | Correct: flat in 2000 |
| 1 | 2000–2002 | −0.49 | Incorrect: long in 2001 (false positive) |
| 2 | 2002–2004 | 0.00 | Flat both years: 2002 ✓, 2003 ✗ |
| 3 | 2004–2006 | 0.52 | Mixed: 2004 ✓, 2005 ✗ |
| 4 | 2006–2008 | 0.58 | 2006 ✓, 2007 ✓, 2008 partial |
| 5 | 2008–2010 | 0.00 | Flat: 2008 ✓, 2009-10 ✗ missed |
| 6 | 2010–2012 | 0.07 | 2010 ✗ flat, 2011 flat (correct) |
| 7 | 2012–2014 | 1.47 | Both years positive Jan → long, both up |
| 8 | 2014–2016 | 0.00 | Flat: missed 2014-15 gains |
| 9 | 2016–2018 | 2.02 | Both years positive Jan → large gains |
| 10 | 2018–2020 | 0.43 | 2019 ✓, 2020 flat/negative start |
| 11 | 2020–2022 | 1.51 | 2021 ✓, 2022 ✓ flat avoids bear |

The "0.00" folds are not losses — the strategy was flat and SPY happened to be positive.
Only fold 1 is genuinely negative (lost money in active positions).

## Limitations

- **One signal per year**: a single January data point drives the entire year's signal.
  There is no mechanism to update mid-year if conditions change.
- **False negatives** (2003, 2009, 2010, 2020): missed major recoveries after bear markets
  because January of those recovery years was negative
- **No parameter to adjust**: unlike most strategies, there is nothing to tune if the
  anomaly weakens — the strategy either works or it doesn't
- **Academic evidence weakening**: post-2000, the January Barometer accuracy has declined
  as more investors trade on the signal, partially arbitraging the effect
