# Set 9 (H601–H620): Regime-Signal Forms, Robustness & Realism

## Thesis
Batches 3–4 established the winning architecture: **VIX-regime-conditional vol-target leverage on the
55-strategy pool**, with VIX<15 the Sharpe optimum (H600/H591: 2.237) and asymmetric tv+cap the Calmar
optimum (H592: 2.264). The regime-leverage frontier is now near-saturated, the holdout is spent, and
further naive Sharpe-chasing risks overfitting. Set 9 therefore pursues two responsible directions:
(1) **genuinely new regime-signal forms** that might add real information, and (2) **robustness/realism
validation** of the champion — the honest tests that matter once a strategy is this good.

Baselines: H600/H591 Sh 2.237; H592 Calmar 2.264; H599 min-fold 1.311. Keep bar: Sharpe>0.51 AND CI-low>0.
All runners use the cached 55-pool (`set8_pool.build_pool`) + `set8_lev` helpers; cache-hit (~seconds each).

---

## Batch A (H601–H605): New regime-signal forms
- **H601 ContinuousVIXLeverage** — cap as a smooth decreasing function of VIX (cap=clip(3.5−VIX/10, 1.0, 3.0)) instead of discrete states. Tests continuity vs the VIX<15 step.
- **H602 VIXPercentileRegime** — bull = VIX below its *expanding* 40th-percentile (adaptive, no fixed level). Robust to secular VIX drift.
- **H603 CompositeRegimeScore** — regime score = mean of {VIX<15, SPY>200d-MA, SPY-TLT 60d corr<0}; cap scales 1.0→2.5 with score. Multi-signal fusion.
- **H604 VIXMomentumRegime** — bull = VIX<15 AND VIX not rising over 20d. De-risk on rising vol even at low levels.
- **H605 RealizedVolRegime** — regime from the portfolio's *own* realized-vol percentile (no external VIX). Tests whether the edge needs VIX or just a vol-state proxy (generalization to VIX-less assets).

## Batch B (H606–H610): Pool & weight robustness
- **H606 ConcentratedPoolVIX15** — VIX<15 regime on the top-12 standalone-Sharpe sub-pool (H516 small-pool finding).
- **H607 ShrinkageWeightsVIX15** — lev-Sharpe + μ·‖w−w_mdr‖² regularization (μ sweep) under VIX<15.
- **H608 EqualWeightVIX15** — equal-weight 55-pool + VIX<15 leverage. No-optimization baseline: if ≈ optimized, weights aren't the edge.
- **H609 SubperiodStableVIX15** — weights optimized on each half (1998–2010, 2010–2022) then averaged. Stability to regime drift.
- **H610 EnsembleOfChampions** — equal-weight blend of the leveraged return streams of H581/H591/H592/H595, then re-vol-targeted. Diversifies across regime configs.

## Batch C (H611–H615): Frontier objectives at the VIX<15 optimum
- **H611 VIX15Asymmetric** — asymmetric tv+cap (H592 mechanism) at threshold VIX<15. Combines the two batch-4 champions.
- **H612 VIX15Calmar** — lev-Calmar objective at VIX<15. Push Calmar past H592 (2.264).
- **H613 VIX15Sortino** — downside-vol leverage at VIX<15. CAGR frontier.
- **H614 VIX15LowCorr** — H001-corr-penalized lev-Sharpe at VIX<15. Lower corr at higher Sharpe than H596.
- **H615 MultiObjectiveVIX15** — maximize Sharpe×Calmar (product) at VIX<15. Balanced champion.

## Batch D (H616–H620): Realism & terminal validation
- **H616 RealBorrowVIX15** — VIX<15 champion + Fed Funds step borrow cost (H588 mechanism). Net-of-financing champion.
- **H617 TurnoverCostVIX15** — add daily leverage-rebalance cost (|Δlev|·1.5bps) to the champion. Capacity/cost realism.
- **H618 RegimeLagVIX15** — lag the VIX regime signal 1/3/5 days. Execution-realism: does the edge survive signal delay?
- **H619 BootstrapRobustVIX15** — bootstrap-averaged weights (100 resamples) at VIX<15 (H599 mechanism on the champion threshold).
- **H620 GrandChampion620** — terminal synthesis: select the best Set-9 mechanism, full battery (5-start stability, sensitivity, comparison vs H600). The program's final strategy.

## Success / falsification
KEEP per the program bar. Beyond that: a Set-9 strategy is *notable* only if it beats the relevant champion
(Sharpe>2.237, Calmar>2.264, or corr<0.299 at Sh>2.0) OR strengthens robustness (H608/H618/H616 degrade < 0.10
Sharpe = champion is real, not fragile). H605/H608 are the key falsification tests: if equal-weight or a
VIX-less vol proxy nearly matches the champion, the "edge" is mostly the leverage mechanism, not the signal/weights.
