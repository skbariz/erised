# Batch 4 (H591–H600): VIX-Regime Synthesis & Frontier Champions

## Thesis (revised from Batch3 results)
Batch3 answered the central question: **a forward-looking VIX regime signal beats the
backward-looking 200d-MA regime.** H581 (VIX<20 → bull cap=2.5×, VIX≥20 → bear cap=1.0×)
is the new program Sharpe champion at **2.119**, beating H570 (MA regime, 2.021) by +0.098.
Secondary winners: H586 (asymmetric tv+cap, 2.054 / Calmar 1.823), H587 (lev-Calmar objective,
Calmar 2.181 / MaxDD −5.41%), H583 (downside-vol, CAGR 17.60%). Pool expansion (H584/H579)
and in-objective MaxDD penalty (H590) both failed.

Batch4 takes the **VIX regime signal as the fixed winning architecture** and (a) tunes/combines it
to push past 2.119, and (b) ports the best Batch3 objectives onto it to claim the remaining
program frontiers (Calmar, CAGR, low-H001-corr, DD-constrained). H600 is the terminal Grand Champion.

All runners use the cached 55-strategy pool (`set8_pool.build_pool()`, `00` cache) and shared
machinery in `scripts/set8_lev.py`. Baseline references: H581 Sh 2.119, H586 2.054, H587 Calmar 2.181,
H583 CAGR 17.60%. Keep bar (CLAUDE.md): Sharpe > 0.51 AND CI-lower > 0.

---

## H591 — VIXThresholdOptimal (+ H581 sensitivity)
Sweep the VIX regime threshold ∈ {15, 17.5, 20, 22.5, 25}, re-optimizing lev-Sharpe weights at each.
Doubles as the required sensitivity analysis for champion H581 (which fixed threshold=20).
**Success:** a threshold beats H581 (2.119) by >0.02 → new champion. **Falsify:** 20 is within ±0.02 of best → H581 robust.

## H592 — VIXAsymmetricVolTarget (combine two best Batch3 ideas)
H581 VIX signal + H586 asymmetric mechanism. Bull (VIX<20): tv=8%/cap=2.5×; Bear (VIX≥20): tv=4%/cap=1.0×.
**Success:** Sh > 2.119 (beats both parents). **Falsify:** Sh < max(H581, H586) → ideas don't compose.

## H593 — VIXThreeStateLeverage
3-state VIX caps: VIX<15 → cap=3.0×; 15≤VIX<25 → cap=2.0×; VIX≥25 → cap=1.0×. tv=6.5%.
**Success:** Sh > 2.119 → finer granularity helps. **Falsify:** ≤ H581 → binary VIX already optimal (cf. H582 ≈ H581).

## H594 — VIXRegimeCalmar (Calmar champion)
H587 leveraged-Calmar objective + VIX regime caps (bull 2.5×/bear 1.0×).
**Success:** Calmar > 2.181 (beats H587) → new Calmar record. Report Sharpe alongside.

## H595 — VIXRegimeSortino (CAGR champion candidate)
H583 downside semi-deviation vol estimate in the leverage + VIX regime caps.
**Success:** CAGR > 17.60% (beats H583) AND MaxDD better than −10.81%. The CAGR frontier with controlled DD.

## H596 — VIXRegimeLowCorr (institutional diversifier)
lev-Sharpe − λ·corr(port, H001) under VIX regime; sweep λ ∈ {0.5, 1.0, 2.0, 3.0}.
**Success:** H001 corr < 0.30 at Sh > 1.80 (H581 is +0.411) → new Sharpe/corr Pareto point.

## H597 — VIXRegimeCAGRMax (family-office mandate)
Leveraged-CAGR objective + VIX regime with bull cap=3.0×.
**Success:** CAGR > 16% at Sh > 1.6. **Guard:** require CI-lower > 1.4 (CAGR-max is overfitting-prone).

## H598 — VIXRegimeDDFloor (DD-constrained, done right)
H590 showed an in-objective MaxDD *penalty* destroys value. Instead: optimize lev-Sharpe, then
deterministically scale the leverage cap down until realized MaxDD ≤ 7% (post-processing, no penalty).
**Success:** Sh > 1.90 at MaxDD ≤ −7%. Tests whether the DD constraint is reachable without the destructive penalty.

## H599 — VIXRegimeBootstrapRobust (overfitting check)
Bootstrap-average SLSQP weights (N=100 resamples of the OOS matrix) for the best mechanism from H591–H593,
under VIX-regime lev-Sharpe. Report worst-fold Sharpe vs H581's min fold.
**Success:** min-fold Sharpe ≥ H581's min AND Sh > 1.95 → robust, not overfit.

## H600 — GrandChampion600 (terminal strategy)
Synthesis of the best regime signal + best objective + final weights. Full report: Sh/CAGR/MaxDD/Calmar/
CI/fold-range/H001-corr, 5-start weight-stability, sensitivity (VIX threshold ±2.5, tv ±1%, cap ±0.5×),
and a comparison table vs H581/H570/H559/H300/H095. The terminal strategy of the 600-hypothesis program.

---

## Execution
Shared helpers in `scripts/set8_lev.py`; each runner imports `build_pool` + helpers and is a thin
objective+leverage spec. All are `00`-cache hits (~seconds each). Commit granularly per hypothesis.
