# H451–H460 Design Doc: DuPont Level Signals & Untested Capital Return Ratios

**Date:** 2026-06-01  
**Set:** 7, Batch 1  
**Status:** IMPLEMENTED — awaiting OOS results

---

## Hypothesis

H201–H250 systematically tested YoY CHANGE versions of DuPont ratios. The LEVEL versions are structurally different signals — persistent competitive position rather than recent improvement. The program's consistent finding that level > change (GP/TA level 0.879 vs change 0.764) predicts these level signals should outperform their change-based counterparts. Three level signals have never been tested in any form: OCF/Revenue, GP/(TA-CA), OCF/(CA-CL).

The momentum-combo template (fundamental level + 6m price momentum) guarantees all-fold coverage at ~Sharpe 0.96 with H001 corr ≈ 0.

---

## Strategies

| H# | Class | Signal | Type | Rebalance |
|----|-------|--------|------|-----------|
| H451 | NetProfitMarginLevelPort | NI/Revenue LEVEL | Standalone | Quarterly |
| H452 | AssetTurnoverLevelPort | Revenue/TA LEVEL | Standalone | Quarterly |
| H453 | ReturnOnEquityLevelPort | NI/BE LEVEL | Standalone | Quarterly |
| H454 | CashProfitMarginPort | OCF/Revenue LEVEL | Standalone (novel) | Quarterly |
| H455 | GrossProfitFixedAssetsPort | GP/(TA-CA) LEVEL | Standalone (novel) | Quarterly |
| H456 | GrossProfitDebtCoveragePort | GP/LTD LEVEL | Standalone (novel vs H237) | Quarterly |
| H457 | CashConversionEfficiencyPort | OCF/(CA-CL) LEVEL | Standalone (novel) | Quarterly |
| H458 | NetMarginMomentumPort | z(NI/Revenue) + z(6m mom) | All-fold combo | Monthly |
| H459 | AssetTurnoverMomentumPort | z(Revenue/TA) + z(6m mom) | All-fold combo | Monthly |
| H460 | ROELevelMomentumPort | z(NI/BE) + z(6m mom) | All-fold combo | Monthly |

## Novelty vs Prior Work

- H451 vs H238: H238 tested YoY NI/Revenue CHANGE (0.696). Level is novel.
- H452 vs H230: H230 tested YoY Revenue/TA CHANGE (0.741). Level is novel.
- H453 vs H221: H221 tested YoY ROE CHANGE (0.717). Level is novel.
- H454: OCF/Revenue not tested in any form.
- H455: GP/(TA-CA) not tested in any form.
- H456 vs H237: H237 tested OCF/LTD (0.729). GP/LTD is novel.
- H457: OCF/(CA-CL) not tested in any form.
- H458–H460: All-fold momentum combos using H451/H452/H453 signals for 12-fold coverage.

## Implementation Notes

- H453 (ROE): negative book equity → NaN → neutral rank (excluded, not top-ranked)
- H456 (GP/LTD): firms with LTD < 1% of TA → NaN → neutral rank (not rewarded for being debt-free)
- H457 (OCF/NWC): negative net working capital (CA ≤ CL) → NaN → neutral rank
- H455 (GP/FixedAssets): negative fixed assets (CA ≥ TA, unusual) → NaN → neutral rank
- All clips applied before sector-neutral ranking to prevent outlier domination

## Success Criteria

- Standalone (H451–H457): Sharpe > 0.51, CI lower > 0
- All-fold combos (H458–H460): All 12 folds active, Sharpe ~0.96, H001 corr < 0.05
- Falsified if: level signal duplicates corresponding change signal (cross-sectional rank corr > 0.90 would indicate same information content — check if Sharpe differs meaningfully from H238/H230/H221)

---

## Results

*To be filled after OOS run.*
