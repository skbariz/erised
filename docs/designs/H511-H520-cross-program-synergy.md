# H511-H520 Design Docs -- Set 7 Batch 7: Cross-Program Synergy Combinations

**Date drafted:** 2026-06-03  
**Batch thesis:** After 510 strategies, the most impactful improvements come from combining the
best discoveries across program sets in ways not previously tested. This batch targets specific
pairings motivated by synthesis, and tests the critical MDR ceiling crack via focused all-fold pools.

---

## H511 -- VolScaledCSStarCombo

**Signal:** Equal-weight blend of the 4 all-fold CS stars (H412, H415, H429, H447) with
portfolio-level vol-scaling overlay: blend_rets * min(1, 8%/portfolio_realized_vol). Monthly.

**Why novel:** H086 applied vol-scaling to H021 (AA: 1.109). H511 applies the same mechanism
to the CS portfolio. If vol-scaling is universal, it should improve CS ensemble Sharpe while
reducing MaxDD from the ~-45% range typical of CS strategies.

**Success criteria:** Sharpe > max(H412, H415, H429, H447). MaxDD < -40%.
**Falsified if:** CS portfolio vol is too stable for vol-scaling to add meaningful protection.

**Results:** TBD

---

## H512 -- SeasonalCSQualityBridge

**Signal:** Nov-Apr: hold H415 (StableMarginMomentum) CS portfolio. May-Oct: hold 100% TLT.
Replaces H078's H021 Nov-Apr allocation with the more powerful all-fold CS quality signal.

**Why novel:** H078 (SeasonalRiskParity, H001 corr +0.008) earns Nov-Apr via H021 and holds TLT
in May-Oct. H512 replaces H021 with H415 (Sharpe 0.987, H001 corr +0.002) in the favorable
season. Should add alpha in Nov-Apr while maintaining the H078 May-Oct defensive leg.

**Success criteria:** Sharpe > H078 (expected ~0.65 baseline). MaxDD < -20%.
**Falsified if:** H415 in Nov-Apr is more volatile than H021 → MaxDD exceeds H078's.

**Results:** TBD

---

## H513 -- QualitySeasonalMarketNeutral

**Signal:** H078's seasonal structure: Nov-Apr = H021 (risk parity). May-Oct = 50% TLT +
50% H473 (LongShortPCAReversal). Adds market-neutral L/S alpha to H078's defensive season.

**Why novel:** H078's May-Oct allocation (100% TLT) has zero equity alpha. H473 is dollar-neutral
(zero equity beta), so adding it during May-Oct should add return without adding equity beta.
The H473 leg provides pure idiosyncratic reversal alpha in the defensive season.

**Success criteria:** Sharpe > H078. MaxDD < H078.
**Falsified if:** H473 in May-Oct adds volatility that offsets TLT's stability.

**Results:** TBD

---

## H514 -- AllFoldMDRPort

**Signal:** MDR optimization restricted to strategies with 12/12 active folds only. Pool:
H001-H100 AA (35 strategies) + Set6 all-fold CS stars (H412, H415, H429, H447) + Set7 all-fold
CS keepers (H473, H499, H505, H508). Total: ~43 strategies.

**Why novel:** The EDGAR gap contaminated all prior MDRs. By enforcing all-fold constraint, the
MDR optimizer works with clean 12-fold return histories for every strategy. The CS additions
(H473 H001 +0.029, H499 H001 +0.004, H505 H001 +0.000, H508 H001 -0.000) are near-zero H001
corr diversifiers that the MDR optimizer has never had available before.

**Success criteria:** Sharpe > H300/H350 (1.525) -- the terminal champion.
**Falsified if:** All-fold restriction doesn't break ceiling -- the frontier is due to genuine
signal correlation among available strategies, not EDGAR contamination.

**Results:** TBD

---

## H515 -- ExpandedAAOnlyMDR

**Signal:** MDR over the full AA strategy pool: H001-H100 (37 AA keepers) + H301-H340 AA
strategies (28 keepers). Total: 65 AA-only strategies. No CS panel required.

**Why novel:** H095 (MDR over 35 H001-H100 strategies: 1.133) is the AA-only benchmark.
H515 tests whether adding Set4's 28 AA strategies to the pool improves the AA MDR frontier.
H350 showed Set4 AA strategies didn't help the combined pool; H515 tests AA-only context.

**Success criteria:** Sharpe > H095 (1.133). MaxDD < H095 (-7.16%).
**Falsified if:** Set4 AA strategies are redundant with H001-H100 AA in MDR space.

**Results:** TBD

---

## H516 -- CoreDiversifierTripleMDR

**Signal:** MDR over a focused 20-strategy pool: top 10 H300 weight contributors (H016, H035,
H043, H060, H065, H068, H076, H078, H086, H097) + 10 CS all-fold stars (H194, H265, H412, H415,
H429, H447, H473, H499, H505, H508).

**Why novel:** The MDR saturation finding: H300's 52 strategies may contain redundancy. H516
tests whether a smaller, curated 20-strategy pool achieves higher MDR Sharpe by eliminating
redundancy dilution. H300's top-10 contributors were selected by MDR itself; adding the 10
best near-zero-H001-corr CS strategies targets the highest-quality pool.

**Success criteria:** Sharpe > H300/H350 (1.525). If 1.53+, confirms small-pool MDR is superior.
**Falsified if:** Sharpe < H300 -- MDR benefits from pool breadth more than pool quality.

**Results:** TBD

---

## H517 -- VolTunedH300Port

**Signal:** H300's MDR portfolio (use H300's stored weights) with daily vol-scaling: portfolio
returns * min(1, 8%/portfolio_realized_vol). Re-run H300's 17 non-zero-weight components to
reconstruct the portfolio return stream, then apply vol-scaling at the portfolio level.

**Why novel:** H086 proved vol-scaling on H021 improves both Sharpe (0.695 -> 1.109) AND MaxDD
(-18.41% -> -9.93%). H517 applies the same mechanism to the H300 base (Sharpe 1.525, MaxDD -6.86%).
Expected: MaxDD should compress below -6.86%; Sharpe should exceed 1.525.

**Success criteria:** Sharpe >= 1.525. MaxDD < -6.86%.
**Falsified if:** H300 portfolio is already so well-diversified that vol-scaling adds no benefit.

**Results:** TBD

---

## H518 -- RatchetH300Port

**Signal:** H300's MDR portfolio with H087's ratchet drawdown protection. When H300 portfolio
drawdown from peak exceeds 5%, scale all positions by (1 - dd/0.15). Same 17-component
reconstruction as H517.

**Why novel:** H087 (ratchet on H021: 1.010, MaxDD -10.93%). Applied to H300 (MaxDD -6.86%),
the ratchet should trigger only in the most severe drawdowns -- preserving returns while providing
emergency protection during rare multi-sigma events.

**Success criteria:** Sharpe >= 1.525. MaxDD < -6.86%.
**Falsified if:** H300 is already so defensively constructed that ratchet merely reduces returns.

**Results:** TBD

---

## H519 -- HierarchicalAllFoldMDR

**Signal:** 2-tier hierarchical MDR.
- Tier 1 (50% weight): MDR over H001-H100 AA pool (35 strategies, same as H095).
- Tier 2 (50% weight): MDR over all-fold CS stars only (H412, H415, H429, H447 + H473, H499, H505, H508).
Final blend: 50% × Tier1_returns + 50% × Tier2_returns.

**Why novel:** H325 (hierarchical MDR, 1.520) used full H150-H199 CS pool in Tier 2, contaminated
by EDGAR gap. H519 restricts Tier 2 to ONLY all-fold CS strategies. Clean, gap-free Tier 2 may
eliminate the ceiling drag that prevented H325 from beating H300.

**Success criteria:** Sharpe >= 1.525. Hierarchical clean > H325 (1.520).
**Falsified if:** Sharpe < H325 -- restricting to all-fold CS reduces Tier2 pool too much.

**Results:** TBD

---

## H520 -- SmallPoolOptimalMDR

**Signal:** MDR over 15 carefully selected strategies: H043, H078, H035, H060, H086, H087,
H097 (7 AA: H300's best + program diversifiers) + H194, H473, H471, H472, H415, H412, H429,
H447 (8 CS: L/S market-neutral + CS all-fold stars).

**Why novel:** MDR saturation at 52 strategies. H520 tests whether 15 high-quality, maximally
decorrelated strategies achieve higher Sharpe than 52. The pool concentrates on: (1) AA
diversifiers with low pairwise correlation, (2) market-neutral L/S with near-zero H001 corr,
(3) all-fold CS with near-zero H001 corr.

**Success criteria:** Sharpe > H300/H350 (1.525). If 1.53+, confirms small-focused MDR is superior.
**Falsified if:** Sharpe < H095 (1.133) -- smaller pool hurts even with best selection.

**Results:** TBD
