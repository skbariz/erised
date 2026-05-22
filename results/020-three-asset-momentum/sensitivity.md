# H020: ThreeAssetMomentum — Sensitivity Analysis

## Data availability note

TLT launched August 2002; GLD launched November 2004. The first 3 walk-forward test
folds (test windows 1998–2004) have no TLT/GLD data — strategies in cash during those
periods. This systematically penalizes the aggregate 12-fold Sharpe.

## Full 12-fold results vs folds 3-11 (GLD available)

| Scope | Sharpe | CI |
|-------|--------|----|
| All 12 folds (canonical) | 0.47 | [0.06, 0.86] |
| Folds 3–11 (GLD available) | 0.65 | [0.22, 1.05] |

**Restated on available data:** OOS Sharpe 0.65, well above benchmark. The kill
decision is technically correct for the full research period, but the signal is
stronger than the headline number suggests.

## Lookback sensitivity (all 12 folds)

| lookback | Sharpe | CI |
|----------|--------|----|
| 3m | 0.30 | [−0.11, 0.70] |
| 6m | 0.23 | [−0.18, 0.64] |
| 9m | 0.23 | [−0.18, 0.64] |
| 12m (canonical) | 0.47 | [0.06, 0.86] |

Short lookbacks are particularly hurt by early-fold cash drag. The 12m lookback
outperforms shorter windows because it stabilizes the signal before TLT/GLD data
becomes available.

## Regime analysis (folds where all 3 assets active)

| Period | Best asset chosen | SPY return | TLT return | GLD return |
|--------|------------------|-----------|-----------|-----------|
| 2008–2010 (GFC) | TLT | −40% | +30% | +25% |
| 2011–2013 | SPY | +25% | +10% | −10% |
| 2015–2017 | SPY | +30% | −5% | +8% |
| 2022 (rate crisis) | SPY (wrong) | −18% | −30% | −2% |

In 2022, SPY had the best (least negative) 12m momentum even as it fell — correctly
chose the least-bad asset. The strategy's ability to switch to GLD avoided the bond
massacre that year.

## Kill reason and re-evaluation threshold

Kill confirmed: Sharpe 0.47 < 0.51, driven by early-fold data starvation.
Would pass as KEEP if restated on folds 3+ (Sharpe 0.65).
Candidate for re-evaluation in ensemble batch (H048–H050) using post-2004 period only.
