# Set 10 (H621–H640): VIX-Momentum Refinement, Leverage Mechanics & Robustness

## Thesis
Set9 found the decisive lever: the **VIX-momentum** regime signal (deploy leverage only when VIX is
low AND not rising). H620 (VIX<16 AND not-rising, 20d) is the terminal champion at **Sharpe 2.472**.
Set10 pushes on the two things not yet exhausted around that signal: (1) **refining the VIX-momentum
signal** itself (threshold × window × direction granularity, rate-of-change), and (2) the **leverage
mechanics** that have been held fixed since H559 — the vol-estimation lookback (always 20d rolling),
the estimator (rolling vs EWMA), and path-dependent throttles (drawdown ratchet, vol-of-vol). Batches C/D
re-map the frontier objectives and re-run the robustness/realism battery on the VIX-momentum champion.

Baselines: H620 Sh 2.472 (VIX<16 not-rising); H604 2.422 (VIX<15 not-rising); H592 Calmar 2.264.
55-pool + `set8_lev`; keep bar Sharpe>0.51 AND CI-low>0. New helper: `set8_lev.vix_momentum_mask`.

---

## Batch A (H621–H625): VIX-momentum signal refinement
- **H621 VIXMomentumGrid** — sweep threshold {15,16,17,18} × not-rising window {10,20,40}; pick best. Sensitivity + optimum for H620.
- **H622 VIXMomentumAsymmetric** — VIX-momentum + asymmetric tv+cap (bull tv=8%/cap=2.5x, bear tv=4%/cap=1.0x). Combine the two strongest levers.
- **H623 VIXRateOfChangeContinuous** — continuous cap = clip(3.0 − k·max(0, VIX 21d %change), 1.0, 3.0); de-risk in proportion to how fast VIX is rising.
- **H624 VIXMomentum3State** — 3-state on (level<16) × (direction): VIX falling→cap 3.0, flat→2.0, rising→1.0.
- **H625 VIXMomentumDownsideVol** — VIX-momentum + downside semi-deviation vol leverage. CAGR frontier on the champion signal.

## Batch B (H626–H630): leverage mechanics (held fixed since H559)
- **H626 LeverageLookbackGrid** — vol-target lookback sweep {10,15,20,30,63} under VIX-momentum. Tests the always-20d assumption.
- **H627 EWMAVolLeverage** — EWMA vol estimate (span 20) instead of trailing rolling std, under VIX-momentum.
- **H628 DrawdownThrottleVIXMom** — multiply the regime cap by a drawdown throttle (1+dd·k on the base portfolio's running drawdown). De-risk in portfolio drawdowns, layered on the regime.
- **H629 RatchetVIXMom** — H087 ratchet (cut exposure proportional to drawdown from peak) layered on VIX-momentum leverage.
- **H630 VolOfVolScaling** — scale leverage by inverse vol-of-vol (VIX's own 20d std percentile); de-risk when volatility is itself uncertain. VIX-momentum base.

## Batch C (H631–H635): frontier objectives at the VIX-momentum optimum
- **H631 VIXMomCalmar** — lev-Calmar objective at VIX-momentum. Push Calmar past H592 (2.264).
- **H632 VIXMomLowCorr** — H001-corr-penalized lev-Sharpe at VIX-momentum (λ sweep).
- **H633 VIXMomSortino** — downside-vol lev-Sharpe at VIX-momentum. CAGR frontier.
- **H634 VIXMomMultiObj** — maximize Sharpe×Calmar at VIX-momentum. Balanced champion.
- **H635 VIXMomDDFloor** — post-hoc tv reduction to MaxDD ≤ 7% at VIX-momentum (no in-objective penalty).

## Batch D (H636–H640): robustness, realism & terminal
- **H636 RealBorrowVIXMom** — VIX-momentum champion + Fed Funds step borrow cost.
- **H637 TurnoverVIXMom** — VIX-momentum champion + leverage-rebalance turnover cost.
- **H638 RegimeLagVIXMom** — lag the VIX-momentum signal 1/3/5 days (execution realism).
- **H639 BootstrapVIXMom** — bootstrap-averaged weights (100 resamples) at VIX-momentum.
- **H640 GrandChampion640** — terminal synthesis: select the best Set10 mechanism, full battery (5-start stability, threshold/window/lookback sensitivity, comparison vs H620). Program's final strategy.

## Success / falsification
KEEP per bar. Notable only if it beats H620 (Sharpe>2.472), H592 (Calmar>2.264), or strengthens robustness.
H626/H627 test whether the fixed 20d-rolling vol estimator was optimal; H628/H629 test whether a path-dependent
throttle adds anything over the regime cap (H557 found ratchet degrades low-DD bases — expect similar). H623's
continuous rate-of-change is the cleanest test of whether "how fast VIX rises" carries more than the binary not-rising flag.
