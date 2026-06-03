# H501-H510 Design Docs -- Set 7 Batch 6: Execution-Aware Cost Minimization

**Date drafted:** 2026-06-03  
**Batch thesis:** H292 (quarterly GP/TA: 0.884) > H227 (monthly GP/TA: 0.852) established that
reducing rebalance frequency improves net Sharpe for slow-moving fundamental signals. Batch 6
extends this to: annual rebalance (the filing-aligned extreme), semi-annual, asymmetric hold
bands, patience entry filters, large-cap filtering, and the 80/20 AA+CS hybrid.

---

## H501 -- AnnualRebalanceQualityPort

**Signal:** GP/TA level, sector-neutral top 20%, annual rebalance (252 trading days).

**Why novel:** H292 quarterly (0.884) > H227 monthly (0.852). Does annual beat quarterly?
GP/TA is an annual filing signal -- annual rebalance perfectly aligns with information frequency.
The IC may hold over 12 months if the signal is structural (persistent quality moats).

**Success criteria:** Sharpe > H292 (0.884).  
**Falsified if:** Annual < quarterly by >0.02 -- information decay before 12 months is real.

**Results:** TBD

---

## H502 -- AnnualRebalanceNetMarginPort

**Signal:** NI/Revenue LEVEL, sector-neutral top 20%, annual rebalance (252 days).

**Why novel:** Tests annual cadence for the new DuPont net margin signal (H451 monthly ~0.692).
NI/Revenue captures full cost-stack efficiency including SGA, interest, taxes. Annual rebalance
aligns with the filing frequency of the signal.

**Success criteria:** Sharpe > H451 (monthly, expected ~0.692).  
**Falsified if:** Annual > monthly confirms signal is structural; annual < monthly confirms
net margin has high rank turnover.

**Results:** TBD

---

## H503 -- SemiAnnualAssetTurnoverPort

**Signal:** Revenue/TA (asset turnover) LEVEL, sector-neutral top 20%, semi-annual (126d).

**Why novel:** H452 tested asset turnover monthly (expected ~0.760). Semi-annual sits between
quarterly and annual. Revenue/TA is mid-speed DuPont component -- faster than structural GP/TA
but slower than earnings or price signals. Optimal cadence is likely between quarterly and annual.

**Success criteria:** Sharpe > H452 (monthly, expected ~0.760).  
**Falsified if:** Semi-annual < quarterly (signal has more rank volatility than quality).

**Results:** TBD

---

## H504 -- LowTurnoverNetMarginPort

**Signal:** NI/Revenue LEVEL, buy top 20%, hold until below 25th percentile. Quarterly rebalance.
Asymmetric entry/exit band: enter when in top 20%, stay until rank drops to bottom 25%.

**Why novel:** H294 applied turnover cap to GP/TA (0.857). H504 applies asymmetric hold band
to NI/Revenue -- a signal with potentially higher rank volatility than GP/TA (NI fluctuates more
than gross profit). Large hold zone (top 20% to 25th percentile) significantly reduces churn.

**Success criteria:** Sharpe > H502 (annual) or H451 (monthly).  
**Falsified if:** Hold band doesn't help -- NI/Revenue rank volatility too high even with wider band.

**Results:** TBD

---

## H505 -- PatienceFilteredMomentum

**Signal:** 6m price momentum (top 20%), BUT only initiate positions on stocks with 3+ consecutive
positive weekly (5-day) returns. Exit when momentum rank drops below 30th percentile. Monthly rebalance.

**Why novel:** All prior momentum variants (H150, H157, H293, H506) rank stocks and enter immediately.
H505 requires CONFIRMATION before entry: 3 consecutive positive weeks ensures momentum is actively
running, not a lagging signal that peaked. Should reduce entry into reverting winners.

**Success criteria:** Sharpe > H157 (vol-scaled momentum: 1.079). MaxDD < H150 (--52.12%).  
**Falsified if:** Patience filter delays entry enough to miss most of the momentum gain.

**Results:** TBD

---

## H506 -- MinTurnoverMomentumBand

**Signal:** 6m momentum, asymmetric band: buy only if ranked above 60th percentile (top 40%),
sell only if ranked below 35th percentile (bottom 35%). Large 25-pct hysteresis zone.
Monthly rebalance.

**Why novel:** H293 (bi-monthly momentum: 0.956) reduced turnover by rebalancing less frequently.
H506 reduces turnover differently: larger entry/exit thresholds at the same monthly cadence.
The asymmetric band prevents exits on normal rank fluctuations while maintaining quality entry bar.

**Success criteria:** Sharpe > H293 (0.956). MaxDD < H293 (--50.24%).  
**Falsified if:** Larger entry threshold (top 40% instead of top 20%) dilutes signal quality enough
to offset turnover savings.

**Results:** TBD

---

## H507 -- LargeCapQualityFilter

**Signal:** GP/TA quality, sector-neutral top 20%, restricted to stocks in top 50% by market cap
within each sector (above sector median market cap). Quarterly rebalance.

**Why novel:** H436 excluded top 10% mega-caps. H507 restricts to top 50% (large caps only),
excluding the smaller half of the S&P 500. Large-cap stocks have better liquidity -- realized
slippage lower than 1.5 bps/turn assumption. Effectively tests if quality signal has higher
net alpha in liquid names.

**Success criteria:** Sharpe > H227 (GP/TA monthly: 0.879).  
**Falsified if:** Large-cap restriction reduces universe enough to hurt diversification.

**Results:** TBD

---

## H508 -- SemiAnnualMomentumQualityCombo

**Signal:** z(GP/TA) + z(6m momentum), sector-neutral top 20%, semi-annual rebalance (126 days).

**Why novel:** H231 QualityMomentum monthly (0.870). H292 showed quarterly beats monthly for GP/TA
alone. H508 tests semi-annual for the composite. The composite has a faster-decaying momentum
component -- optimal frequency may be higher than annual but lower than monthly.

**Success criteria:** Sharpe > H231 (0.870).  
**Falsified if:** Sharpe < 0.820 -- composite decay at lower frequency too severe.

**Results:** TBD

---

## H509 -- LowTurnoverROEPort

**Signal:** NI/BE (return on equity) LEVEL, buy top 20%, hold until below 25th percentile.
Quarterly rebalance. Negative book equity stocks excluded (assigned NaN signal).

**Why novel:** H453 tested ROE level monthly (expected ~0.751). H509 applies asymmetric hold band
to ROE. ROE includes leverage (unlike GP/TA), so its cross-sectional rank may be more stable for
firms with persistent high ROE (structural advantages). Low-turnover may improve it significantly.

**Success criteria:** Sharpe > H453 (monthly, expected ~0.751).  
**Falsified if:** ROE rank volatility too high for the hold band to reduce turnover meaningfully.

**Results:** TBD

---

## H510 -- CrossProgramHybridLowTurnover

**Signal:** 80% H021 (RiskParity, daily ETF rebalance) + 20% H415 (StableMarginMomentum, quarterly
CS rebalance 63d). Explicit frequency-by-asset-class separation: ETFs need no minimum frequency;
stocks benefit from low turnover at the CS layer.

**Why novel:** H490 tested 50/50 H086+H415 (both at different frequencies). H510 tests 80/20
with explicit quarterly CS layer -- a more conservative hybrid that preserves H021's core
risk-parity diversification while adding a small orthogonal CS return stream at low turnover.

**Success criteria:** Sharpe > H021 (0.695). MaxDD < H021. H001 corr < H086 (decorrelated by CS).  
**Falsified if:** 80% H021 dominates the blend and the 20% CS adds no meaningful alpha.

**Results:** TBD
