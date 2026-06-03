# H521-H530 Design Docs -- Set 7 Batch 8: Novel Fundamental Combinations

**Date drafted:** 2026-06-03  
**Batch thesis:** Novel signals from available Compustat data not tested in any prior hypothesis.
All are CS strategies using the existing fundamental panel. H521-H529 are fundamental signals
with EDGAR gap in folds 0-5; H530 is pure price (all 12 folds).

---

## H521 -- NetDebtQualityPort

**Signal:** Net leverage = (LTD - CA) / TA. Long lowest net leverage (most net cash). Sector-neutral top 20%, quarterly.

**Why novel:** H246 tested TA/BE gross leverage (0.759). H237 tested OCF/LTD coverage (0.729). Net debt ratio (LTD minus liquid current assets) is the TRUE economic leverage. A firm with $1B LTD and $2B CA has NEGATIVE net debt (net cash) despite having gross leverage. Favors firms with financial fortress quality.

**Success criteria:** Sharpe > 0.51, CI lower > 0.
**Falsified if:** Net leverage duplicates gross leverage signal (cross-sectional corr > 0.85).

**Results:** TBD

---

## H522 -- CapitalIntensityAvoidPort

**Signal:** (TA - CA) / Revenue = fixed asset intensity. Long lowest intensity (asset-light). Sector-neutral top 20%, ascending, quarterly.

**Why novel:** H452 tested GP/(TA-CA) = output per fixed capital (efficiency). H522 tests (TA-CA)/Revenue = fixed capital input per revenue dollar (input intensity). Tests the INPUT side of the capital efficiency equation. Asset-light firms (software, services) generate more revenue per unit of fixed capacity.

**Success criteria:** Sharpe > 0.51, CI lower > 0.
**Falsified if:** Capital intensity is dominated by sector membership (manufacturing vs software).

**Results:** TBD

---

## H523 -- GrossCrossEfficiencyPort

**Signal:** (GP/TA) x (Revenue/TA) = GP*Revenue/TA^2. Long top 20%. Sector-neutral, quarterly.

**Why novel:** GP/TA (H227: 0.879) and Revenue/TA (H452 new: ~0.760) are tested separately. Their PRODUCT rewards firms that excel on BOTH gross profitability AND asset efficiency simultaneously -- not just their average. Equivalent to gross profit per unit of TA squared.

**Success criteria:** Sharpe > max(H227, H452). Cross-efficiency should outperform single dimensions.
**Falsified if:** Product merely amplifies GP/TA signal (cross-sectional corr > 0.90 with GP/TA alone).

**Results:** TBD

---

## H524 -- CashCycleEfficiencyPort

**Signal:** OCF / (CA - CL) = operating cash flow per unit of net working capital. Long top 20%. Sector-neutral, quarterly. Firms with CA < CL (negative NWC) excluded (NaN).

**Why novel:** Not tested in any form. Distinct from OCF/Revenue (margin) and OCF/TA (return on assets). Measures how efficiently the operating cycle converts working capital to cash. High ratio = lean operations with minimal capital trapped in receivables/inventory.

**Success criteria:** Sharpe > 0.51, CI lower > 0.
**Falsified if:** Firms with negative NWC (retailers with fast-paying customers) dominate the top ranking spuriously.

**Results:** TBD

---

## H525 -- SectorNeutralizedROEPort

**Signal:** NI/BE - sector_median(NI/BE) = sector-demeaned ROE. Long highest sector-adjusted ROE. Sector-neutral top 20%, quarterly.

**Why novel vs H453 (ROE level: ~0.751):** H453 ranks by ROE within sector (removes sector effects via ranking). H525 uses sector-DEMEANED ROE -- an explicit transformation that removes structural leverage differences across sectors (financials have naturally high ROE from leverage). A tech firm at ROE=25% in a 20%-median sector is more exceptional than a financial at ROE=25% in a 22%-median sector.

**Success criteria:** Sharpe > H453 (expected ~0.751).
**Falsified if:** Sector-demeaning doesn't add over standard sector-neutral ranking.

**Results:** TBD

---

## H526 -- ProfitabilityConsistencyPort

**Signal:** Fraction of trailing 8 quarters (504 days) where NI/BE > sector median NI/BE. Long top 20% most consistently above-sector-median ROE. Sector-neutral, quarterly.

**Why novel:** H225 tested earnings GROWTH persistence (positive NI growth quarters). H526 tests ROE LEVEL persistence ABOVE SECTOR BENCHMARK -- structural competitive advantage. A firm consistently beating its sector peers has demonstrated durable moat, not just recent improvement.

**Success criteria:** Sharpe > 0.51, CI lower > 0.
**Falsified if:** Persistence measure is too noisy with quarterly-updated data (many ties).

**Results:** TBD

---

## H527 -- RevenueMarginConsistencyPort

**Signal:** Fraction of trailing 8 quarters where both: (1) quarterly Revenue growth > 0 AND (2) quarterly GP/Revenue change > -1% (no margin deterioration). Long top 20% consistent quality growers. Sector-neutral, quarterly.

**Why novel:** Tests a JOINT persistence condition -- both sustained revenue growth AND margin stability simultaneously. H411-H415 tested these signals separately. Firms that maintain both growth AND margin for 8 consecutive quarters have a reliably compounding business model.

**Success criteria:** Sharpe > 0.51, CI lower > 0.
**Falsified if:** Joint condition too restrictive -- too few stocks pass both criteria simultaneously.

**Results:** TBD

---

## H528 -- OCFRevenueGrowthCombo

**Signal:** z(OCF/Revenue level) + z(Revenue YoY growth). 2-factor equal z-score. Quarterly.

**Why novel:** H233 tested z(rev_growth) + z(GP/TA): 0.803. H528 replaces GP/TA with OCF/Revenue -- cash margin + revenue growth. OCF/Revenue captures cash conversion stack (post-working-capital) unlike GP/Revenue (pre-SGA). The OCF/Revenue + growth combo rewards firms with both real cash generation AND top-line expansion.

**Success criteria:** Sharpe > H233 (0.803).
**Falsified if:** OCF/Revenue is redundant with GP/Revenue at z-score level.

**Results:** TBD

---

## H529 -- NetMarginAssetTurnoverCombo

**Signal:** z(NI/Revenue level) + z(Revenue/TA level). 2-factor DuPont combo. Quarterly.

Note: NI/TA = (NI/Revenue) x (Revenue/TA) mathematically. But ranking on BOTH z-scores separately rewards firms excelling on BOTH margins AND efficiency -- not just firms with high NI/TA via either dimension alone.

**Success criteria:** Sharpe > H173 (ROA: 0.785) -- if DuPont decomposition adds over the product.
**Falsified if:** Sharpe < H173 -- the product (NI/TA) summarizes the information better than the sum of parts.

**Results:** TBD

---

## H530 -- MultiPeriodMomentumBlend

**Signal:** Equal-weight average of z(3m return 63d), z(6m return 126d), z(12-1m momentum skip-1m). Long top 20% by blended multi-horizon momentum. Sector-neutral, monthly.

**Why novel:** All prior momentum strategies (H007, H150, H216) use a SINGLE formation period. H530 selects stocks that rank highly ACROSS ALL TIME HORIZONS -- a multi-horizon consensus filter. Reduces momentum crash risk from any single formation period while maintaining the strongest trend confirmers. Pure price signal, full 12-fold coverage.

**Success criteria:** Sharpe > H157 (vol-scaled momentum: 1.079). MaxDD < H150 (-52.12%).
**Falsified if:** Blending horizons adds noise (averaging weak and strong signals) rather than consensus filtering.

**Results:** TBD
