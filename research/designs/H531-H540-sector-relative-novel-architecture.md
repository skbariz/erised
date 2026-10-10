# H531-H540 Design Docs -- Set 7 Batch 9: Sector-Relative Signals & Novel Portfolio Architecture

**Date drafted:** 2026-06-03  
**Batch thesis:** After 530 strategies, this batch tests: (1) sector-relative signal construction
not previously tried (H531 GP/TA delta, H534 acceleration), (2) novel portfolio architecture
(H533 per-stock vol-scaling, H537 CS ratchet), (3) combined reversal+quality signals (H538,H540),
and (4) filtered universe approaches (H536 quality gate, H539 sector leader filter).

---

## H531 -- SectorDeltaQualityPort

**Signal:** GP/TA minus sector median GP/TA = sector-relative quality premium. Long top 20%
with largest positive deviation from sector median. Sector-neutral top 20%, quarterly.

**Why novel vs H227 (sector-neutral GP/TA: 0.879):** H227 ranks by GP/TA within sector (selects
top 20% within each sector regardless of absolute deviation). H531 uses ABSOLUTE DISTANCE above
sector median -- a firm that is MUCH better than its sector gets a higher score than one barely
above median. Firms with sector-delta > 0 may span multiple sectors (not just top 20% per sector).

**Success criteria:** Sharpe > H227 (0.879) or at least > 0.51.
**Falsified if:** Sector-delta GP/TA ≈ sector-neutral ranking (corr > 0.90 cross-sectionally).

**Results:** TBD

---

## H532 -- TwoStageQualityFilter

**Signal:** Stage 1 → filter to top 40% by GP/TA (sector-neutral). Stage 2 → from filtered set,
rank by NI/Revenue level, take top 20%. Long the double-quality winners. Quarterly.

**Why novel:** H265 (SustainabilityFilteredProfitability: 0.952) filtered GP/TA by accruals gate.
H532 filters GP/TA by NI/Revenue (two quality dimensions in sequential filtering). The filter+rank
approach was the best strategy in Set 3 (H265). H532 tests two quality fundamentals in sequence.

**Success criteria:** Sharpe > H265 (0.952). Two-stage filter should outperform either dimension alone.
**Falsified if:** Sharpe < H227 (0.879) -- filtering to top 40% GP/TA then ranking by NI/Revenue
concentrates in small universe without meaningful improvement.

**Results:** TBD

---

## H533 -- VolScaledMomentumQuality

**Signal:** z(GP/TA) + z(6m momentum), sector-neutral top 20%, with per-stock vol-targeting:
each stock's weight scaled by min(1, 15%/stock_252d_realized_vol). Renormalized. Monthly.

**Why novel:** H157 (VolScaledMomentum: 1.079) applied portfolio-level vol scaling to momentum.
H511 applied portfolio-level vol scaling to CS star ensemble (1.137). H533 applies STOCK-LEVEL
vol scaling to the quality-momentum composite (H231 framework). Per-stock scaling overweights
low-volatility quality names and underweights lottery-like high-vol names.

**Success criteria:** Sharpe > H231 (quality+momentum monthly: 0.870). MaxDD improvement.
**Falsified if:** Per-stock vol scaling adds noise (high-vol quality names are falsely discarded).

**Results:** TBD

---

## H534 -- GPTAAccelerationPort

**Signal:** Second derivative of GP/TA: Δ(ΔGP/TA) over trailing 3 quarters (first deriv quarterly,
second deriv another quarter later). Long top 20% stocks with fastest-accelerating GP/TA. Quarterly.

**Why novel:** H243 tested GP dollar growth (first derivative: 0.791). H534 tests GP/TA ACCELERATION
(second derivative) -- the rate of change of the rate of change. Firms where quality improvement
is itself accelerating are at an inflection point not yet priced. The second derivative is a
leading indicator of future quality trajectory.

**Success criteria:** Sharpe > 0.51. Should exceed H243 GP growth (0.791) since acceleration
is the higher-order signal.
**Falsified if:** Second derivative is too noisy for quarterly-updated fundamentals.

**Results:** TBD

---

## H535 -- LowBetaQualityMomentum

**Signal:** z(-252d beta) + z(GP/TA level) + z(6m momentum). Equal 3-factor composite. Monthly.

**Why:** H178 (BAB + momentum: 1.125) was the best 3-component composite. H535 adds GP/TA quality
alongside BAB and momentum. 3-factor dilution warning (Set 3 finding: composites above 2 factors
typically dilute). But H178 showed BAB+momentum is highly synergistic -- quality may add the third
leg by filtering out leveraged momentum stocks.

**Success criteria:** Sharpe > H178 (1.125). All 12 folds active (price-based BAB + momentum).
**Falsified if:** Sharpe < H178 -- quality dilutes the BAB+momentum signal.

**Results:** TBD

---

## H536 -- EarningsQualityGatedMomentum

**Signal:** 6m price momentum, sector-neutral, filtered: only stocks where OCF>0 AND NI>0 AND
OCF/NI>0.5 (cash earnings backing). Top 20% momentum within quality-gated universe. Monthly.

**Why novel:** H215 FundamentalReversal gated reversal by quality (NI>0 AND OCF>0: 0.718). H536
applies stricter quality gate (OCF/NI > 0.5 = more cash than accrual earnings) to MOMENTUM.
Gate ensures momentum runs in companies with real cash backing -- avoiding momentum crashes in
low-quality names that reverse sharply after earnings disappointments.

**Success criteria:** Sharpe > H150 (momentum: 0.981). MaxDD < H150 (-52.12%).
**Falsified if:** Quality gate reduces universe too much (< 50 eligible stocks per sector).

**Results:** TBD

---

## H537 -- MomentumWithPortfolioRatchet

**Signal:** H150 (6m momentum, top 20%, sector-neutral, monthly) with H087-style portfolio
ratchet: when CS momentum portfolio drawdown from peak exceeds 10%, scale all positions by
(1 - drawdown/0.20). Resets at new high-water mark. Pure price signal, all 12 folds.

**Why novel:** H087 applied ratchet to H021 AA (1.010, MaxDD -10.93%). Momentum crashes (2009,
2020 reversal) are the primary risk in the momentum family. H537 tests whether the ratchet
mechanism reduces momentum portfolio MaxDD while preserving Sharpe.

**Success criteria:** Sharpe > H157 (vol-scaled momentum: 1.079). MaxDD < H157 (-25.39%).
**Falsified if:** Ratchet merely delays recovery -- momentum portfolio drawdowns are too swift
for end-of-day ratchet to protect.

**Results:** TBD

---

## H538 -- ReversionQualityCombo

**Signal:** z(5-day within-sector reversal) + z(GP/TA). Equal weight. WEEKLY (5-day) rebalance.
Long quality stocks temporarily dislocated by short-term price reversals.

**Why novel:** All prior reversal strategies use price only (H185/H186/H189). H538 adds GP/TA
quality as a co-signal -- selects quality firms temporarily dislocated (intersection of "buy the
dip" and "quality matters"). The quality component acts as a persistent filter preventing entry
into low-quality reversal names that may not recover.

**Success criteria:** Sharpe > H185 (sector reversal long-only: 1.103).
**Falsified if:** Quality+reversal combo reduces effective trade count too much vs pure reversal.

**Results:** TBD

---

## H539 -- SectorLeaderMomentumFilter

**Signal:** (1) Compute 6m quality-adjusted momentum per sector: mean(6m_return × GP/TA_zscore)
across sector members. (2) Select top 2 sectors by this metric. (3) Within those 2 sectors, take
top 20% by standard 6m momentum. Monthly.

**Why novel:** H203 (SectorLeaderLong-only: 0.915) selected sectors by 6m return alone. H539
uses QUALITY-ADJUSTED momentum for sector selection -- sectors where both momentum AND quality
are strong. Tests whether fundamental-filtered sector rotation is superior to price-only.

**Success criteria:** Sharpe > H203 (0.915). Should benefit from quality filter in sector rotation.
**Falsified if:** Concentrating in 2 sectors creates excessive volatility vs broader universe.

**Results:** TBD

---

## H540 -- FundamentalReversalWeekly

**Signal:** z(5-day within-sector idiosyncratic reversal) + z(-accruals). Equal weight. Weekly.
Low-accrual stocks that experienced within-sector reversal = quality at a discount.

**Why novel:** H215 FundamentalReversal used MONTHLY reversal gated by quality (0.718). H540
uses WEEKLY reversal + accruals quality as EQUAL z-score (not a gate). Accruals = (NI-OCF)/TA
updates quarterly (forward-filled). Weekly reversal provides the high-frequency signal; negative
accruals (cash-backed earnings) provides persistent quality filter. Distinct from H538 (GP/TA
quality, not accruals).

**Success criteria:** Sharpe > H185 (sector reversal: 1.103). Accruals should improve vs pure reversal.
**Falsified if:** Accruals (EDGAR gap in folds 0-5) degrades performance in early folds.

**Results:** TBD
