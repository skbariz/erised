# Project
Python research repo for S&P 500 trading strategies. Erised trading models.

Walk-forward OOS methodology: research period 1993-2022, holdout sealed (spent after H100).
Folds: 5-year train / 2-year test, 12 non-overlapping folds.
Costs: 0.5 bps commission + 1.0 bps slippage.
SPY benchmark: Sharpe 0.51 | CAGR 8.35% | Max DD -55.92%.

# Repo structure
- Strategy classes: src/quant/strategies/base.py (all strategies in one file)
- Backtest engine: src/quant/backtest/engine.py
- Walk-forward: src/quant/validation/walk_forward.py
- Per-hypothesis runners: scripts/run_h<NNN>.py
- JSON summary (read this first): results/<NNN>-<name>/summary.json
- Full stats: results/<NNN>-<name>/stats.md
- Sensitivity: results/<NNN>-<name>/sensitivity.md (not always present)
- Master leaderboard: results/SUMMARY.md

# Commands
- Install: uv sync
- Test: uv run pytest -x -q
- Lint: uv run ruff check . && uv run ruff format .
- Type check: uv run pyright <file>
- Run hypothesis: uv run python scripts/run_h<NNN>.py
- Sanity check: uv run python scripts/sanity_check.py

# Result file reading order
1. results/<NNN>-<name>/summary.json -- compact JSON, read this first always
2. results/SUMMARY.md -- master leaderboard for cross-strategy context
3. results/<NNN>-<name>/stats.md -- full markdown detail only if needed
4. results/<NNN>-<name>/sensitivity.md -- parameter robustness only if asked

Never open equity curve files, raw trade logs, or parquet files unless explicitly asked.

# Folder naming convention
Results folders use short names: results/021-risk-parity, results/007-ts-momentum.
Not the full strategy description. Check actual folder names with rg or dir before referencing.

# Current program state (PROGRAM COMPLETE — H001-H100)
- Strategies tested: 100
- Kept: 47 total
  Individual (38): H001 H007 H008 H016 H018 H019 H021 H022 H023 H025 H027 H031 H034 H035 H043 H046
                   H051 H060 H063 H065 H066 H068 H070 H076 H077 H078 H079 H080 H082 H084 H086 H087
                   H088 H089 H090 H097 H098
  Ensemble (9): H048 H049 H050 H091 H092 H093 H094 H095 H099
- Program champion: H095 (MaxDiversificationPortfolio) — Sharpe 1.133, Max DD -7.16%
- Best individual strategy: H086 (LowVolTargetPortfolio) — Sharpe 1.109, Max DD -9.93%
- Star strategy (original): H021 (RiskParity) — Sharpe 1.029, Max DD -18.41%
- Lowest H001 corr (kept): H078 (SeasonalRiskParity) +0.008, H098 (ReversionEntryRiskParity) +0.196
- Holdout result (H100): H049 Sharpe 1.346 / Max DD -9.91% vs SPY 1.465 / -18.76% (2023-2026)
  Holdout seal is now SPENT — do not run load_holdout() again for research purposes.
- Goal trackers: ALL MET — Sharpe ≥ 1.0 (H095: 1.133), Max DD < SPY (H095: -7.16%), tested ≥ 100

---

# Strategy family performance summary (H001–H100)

What each family produced and why — distilled from the full 100-strategy program.

## Trend-following (H001, H007, H008, H009–H011, H013–H015, H016, H017, H027)
- Long-only works (H001: 0.61, H007: 0.62, H008: 0.62). Long/short universally fails — shorting adds
  whipsaw costs (1.5 bps/turn) without enough bear-market alpha to compensate.
- Vol-scaling (H016) improves Max DD to -26% but not Sharpe.
- All trend signals have H001 corr > 0.70 — measuring the same underlying signal.
- **Family saturated. Do not pursue more trend variants.**

## Multi-asset rotation (H018–H022, H051–H060)
- H021 (RiskParity) is the foundation of the entire program.
- Rotation strategies (H018, H019, H060) kept but correlated with H001 (~0.29–0.79).
- TLT from 2002, GLD from 2004, DBC from 2006, HYG from 2007. Strategies using late-starting assets
  incur 3-5 zero OOS folds and almost always fail the keep bar.
- H051 (3-asset risk parity): Kept but MaxDD worse (-47.67%) than H021 due to GLD early-fold gaps.
- H060 (Carry+momentum SPY/TLT): First kept strategy below H001 corr 0.30 (+0.294).
- **Lesson:** More assets ≠ better. The GLD/DBC data-availability penalty typically outweighs any
  diversification benefit unless the asset has data from ≥ 2000.

## Calendar effects (H023–H027, H043–H047, H071–H075, H078)
- Confirmed anomalies: Sell-in-May, January Barometer, Halloween effect all pass the keep bar.
- H001 corr range: 0.33–0.57. Best low-corr calendar family.
- H043 (SellInMayBondConfirm): -10.11% MaxDD — first MaxDD record holder.
- H078 (SeasonalRiskParity): H001 corr +0.008 (program all-time record). H021 in Nov-Apr,
  100% TLT in May-Oct. Near-zero correlation with equity momentum — invaluable ensemble component.
- H071-H075 (calendar extensions): All killed. H071 missed by 0.002, H072 had -9.82% MaxDD
  but couldn't deliver Sharpe. Calendar signal saturation — no more alpha from seasonal timing alone.
- **High-value for ensemble selection. H078 and H043 are irreplaceable diversifiers.**

## VIX-based and vol regime (H028–H032, H066–H070)
- H031 (RealizedVsImpliedVol): Only pure VIX-based keeper. Vol risk premium is durable.
- H066 (VIX risk parity enhanced): Sharpe 0.990 via VIX>35 crisis override to 80/20 TLT/SPY.
- H070 (Multi-asset dynamic vol target): Sharpe 1.066, MaxDD -12.28%. New program star.
  H021 weights × (min(1, target_vol/realized_vol)) — reduces exposure when market is stressed.
- H067, H069: VIX-based regime filters killed (CI contains zero or barely positive; fold variance extreme).
- **VIX as regime indicator works; VIX as timing signal fails.**

## Cross-asset macro / regime (H033–H037, H053–H059, H076–H080, H097–H098)
- Bond trend (H034) and gold fear gauge (H035) kept (H001-H050 batch).
- H053 (credit spread), H054 (yield curve), H056 (yield curve rotation): All killed due to data gaps
  and CI-clips-zero. The 42d lookback versions of H054/H056 show promise but are post-OOS params.
- H076 (4-asset parity SPY+TLT+GLD+DBC): Kept, 0.666. Folds 0-2 zero (pre-DBC).
- H077 (risk parity + momentum filter): Kept, 0.683. Momentum gate adds value.
- H079 (crisis protocol risk parity): Sharpe 0.952. Drawdown+VIX hysteresis works.
- H080 (yield curve overlay): Sharpe 0.891. SHY>TLT inversion as SPY-weight reducer is effective.
- H097 (correlation risk parity hybrid): Sharpe 0.992. Best regime-overlay approach:
  positive 60d SPY-TLT rolling correlation → crisis override. Signal is always available, never gaps.
- H098 (reversion entry risk parity): Sharpe 0.879, H001 corr +0.196. Drawdown-triggered entry
  to heavy TLT. Low corr because it fires only in crashes.
- **Regime overlays on H021 consistently produce Sharpe 0.85-0.99. The correlation regime (H097)
  is the most reliable signal — it's always computable and has no data-gap risk.**

## Equity factors (H061–H065)
- H063 (Defensive sector overlay, XLP/XLU via VIX+MA regime): Kept, 0.577.
- H065 (Equal-weight defensive factor, SPY+XLP+XLU+IVE): Kept, 0.554.
- H061 (value/growth rotation), H062 (small-cap), H064 (tech/defensive momentum): All killed.
- MaxDD poor for kept strategies (-39% to -49%) — equity factors provide no downside protection.
- **Equity factors add little beyond diversification noise. Not worth the complexity or MaxDD cost.**

## Vol-control and drawdown management (H086–H090)
- THE most important family discovered in H051-H100. All five strategies kept; four above Sharpe 0.87.
- H086 (LowVolTargetPortfolio): Sharpe 1.109, MaxDD -9.93%. Program Sharpe record (individual).
  Mechanism: H021 × min(1, 8%/realized_vol). Simple and powerful.
- H087 (RatchetDrawdownProtection): Sharpe 1.010, MaxDD -10.93%.
  Cuts exposure proportional to portfolio drawdown depth.
- H088 (H001+H021 50/50 blend): Sharpe 0.871. High H001 corr (+0.924) limits usefulness.
- H089 (H043+H021 50/50 blend): Sharpe 1.074, MaxDD -9.62%. Season+parity blend is excellent.
  Low-corr components blend to sub-component MaxDD; both components needed.
- H090 (Monthly rebalancing risk parity): Sharpe 1.020. Identical to H021 but rebalances monthly
  (N=231 trades vs H021's daily). Robust finding: rebalance frequency doesn't matter much.
- **Key lesson: Position-sizing overlays (vol-target, ratchet) reliably improve both Sharpe AND MaxDD
  simultaneously. This is the primary lever for improving on H021.**

## Contrarian / mean reversion (H038–H042, H083)
- ALL killed. Lowest H001 corr in the program (+0.073 to +0.165) but N=6-18 trades.
- **Do not pursue more contrarian signals** without a mechanism to increase trade frequency.

## Ensemble construction (H048–H050, H091–H095, H099)
- **Method ranking (Sharpe):**
  MDR (H095: 1.133) > H049+H078 (H094: 1.084) > Min-corr v2 (H091: 0.984) >
  Regime-adaptive (H092: 0.964) > Batch19 min-corr (H099: 0.958) > H049 (0.959) >
  Anti-momentum (H093: 0.852) > Sharpe-weighted (H050: 0.820) > Equal-weight (H048: 0.800)
- H095 (MaxDiversificationPortfolio, SLSQP MDR): Sharpe 1.133, MaxDD -7.16% — program champion.
  Concentrates on H043 (24%), H078 (20%), H035 (15%), H060 (13%), H025 (10%).
  The MDR optimizer can assign fractional weights to calendar strategies that binary selection ignores.
- H094 (H049+H078): Sharpe 1.084. Adding H078's zero-corr seasonal component to H049 is the
  simplest ensemble improvement — one additional component, major MaxDD reduction.
- Anti-momentum rebalancing (H093) destroys value: chasing recent underperformers punishes good
  strategies when they have normal drawdowns. Do not implement "revert to mean" ensemble weighting.
- Regime-adaptive weighting (H092) is marginal over equal-weight: daily regime signals are too noisy
  to outweigh the variance penalty of unequal weights.
- **The single most impactful ensemble insight: H078 (zero-corr seasonal parity) is a universal
  diversifier. Adding it to any ensemble reduces MaxDD substantially at minimal Sharpe cost.**

---

# Definitive synthesis: what works in this market regime (1993-2026)

## The dominant pattern
H021 (risk parity) is the bedrock. Everything else is a modification of it.
The best modifications, ranked by marginal Sharpe gain over H021 (1.029):
1. Vol-scaling position sizing (H086): +0.080 Sharpe, -8.48% MaxDD
2. Ratchet drawdown protection (H087): −0.019 Sharpe, -7.48% MaxDD
3. MDR ensemble over 35 strategies (H095): +0.104 Sharpe, -11.25% MaxDD (ensemble context)
4. Seasonal overlay H043+H021 (H089): +0.045 Sharpe, -8.79% MaxDD
5. Correlation crisis override (H097): −0.037 Sharpe, -0.07% MaxDD (vs H021)

## The H001 correlation floor
- All 47 kept strategies have H001 corr between +0.008 (H078) and +0.924 (H088).
- The floor for strategies that trade actively (>50 trades/30yr) is approximately +0.19 (H098).
- Below +0.30: H078 (+0.008), H098 (+0.196), H079 (+0.254), H080 (+0.257), H060 (+0.294).
- Strategies with H001 corr < 0.20 are the most valuable ensemble diversifiers.

## Data availability constraints (immutable)
- TLT: available from 2002 (folds 0-1 in 1993-2002 train period still use SPY-only)
- GLD: available from 2004 (fold 2 test starts 2002 — GLD gaps are in train, not test)
- DBC: available from 2006 → 3-4 zero OOS folds → almost always kills strategies
- HYG/LQD: available from 2007 → same problem
- VIX: no holdout parquet file (VIX_holdout.parquet does not exist)
- Holdout parquet files available: SPY, TLT, GLD, SHY, IEF, LQD, HYG, DBC, VNQ, IVE, IVW, IWM, QQQ, XLP, XLU

---

# Trading rules
- Always report: OOS Sharpe, 95% CI, CAGR, Max DD, H001 corr, trade count, fold Sharpe range.
- A strategy is only "kept" if: CI lower bound > 0 AND OOS Sharpe > SPY benchmark (0.51).
- Never use today's S&P 500 membership for historical backtests. Use point-in-time membership.
- Prefer Polars, DuckDB, and Parquet for data work.
- Sensitivity analysis (1-D parameter sweeps) is required for any kept strategy before finalizing.
- Report per-fold Sharpe for every run - fold variance is signal for overfitting.

# Bias audit checklist (run mentally before any new strategy)
- Lookahead: signals use only data available at decision time? Check .shift(-n), same-bar close used for entry.
- Survivorship: using point-in-time S&P 500 membership?
- Leakage: target variable not derived from features?
- Cost timing: costs applied at next-bar open, not current bar close?
- Parameter snooping: canonical params chosen before OOS, not tuned on OOS?
- The framework shifts positions by 1 bar - verify new strategies respect this convention.

# Context discipline
- Read results/<NNN>-<name>/summary.json before any discussion of a specific strategy.
- Read results/SUMMARY.md before any cross-strategy discussion or ranking.
- Use ripgrep (rg) over reading whole files for code search.
- Use the built-in Explore agent for investigations spanning many files.
- After editing any Python file: uv run ruff format <file> and uv run ruff check <file> - file only.
- If context goes above 60%, suggest /compact to the user before continuing.
- When switching to an unrelated task, suggest /clear instead of continuing in context.

# Session discipline
- Default model: Sonnet 4.6 at medium effort.
- Drop to Haiku for: trivial lookups, one-line edits, simple formatting fixes.
- Escalate to high effort or /advisor for: new strategy design, ensemble construction, hard bias questions.
- Do not spawn a subagent for tasks under 3 file reads or under 5 minutes of work.

# Design-doc discipline
For any new hypothesis:
1. Before coding, write a one-paragraph hypothesis doc at docs/designs/H<NNN>-<name>.md:
   - What signal are you testing and why should it work?
   - Expected behavior in key regimes (2000-2002 tech crash, 2008-2009 GFC, 2020 COVID, low-vol 2013-2017)
   - Success criteria: what Sharpe/CI/H001-corr would make this worth keeping?
   - What would falsify this hypothesis?
2. Implement strategy class in src/quant/strategies/base.py following existing patterns.
3. Write runner scripts/run_h<NNN>.py following run_h021.py as template (includes write_summary call).
4. After running OOS backtest, append results and kill/keep decision to the design doc.

# Coding discipline

## Think before coding
- State assumptions explicitly. If uncertain about methodology, ask rather than guess.
- If a request could mean two different things, name both interpretations and confirm before implementing.
- Push back if a simpler approach exists or if the request risks introducing bias.
- Stop if confused about OOS protocol or the position-shift convention.

## Simplicity first
- Write the minimum code that tests the hypothesis. Nothing speculative.
- New strategy classes go in src/quant/strategies/base.py alongside existing ones.
- New runners follow run_h021.py exactly - same structure, same write_summary call at the end.
- If 200 lines could be 50, write 50.

## Surgical changes
- Touch only what the task requires. Do not refactor base.py while adding a new strategy.
- Match existing style - look at nearby strategy classes before writing a new one.
- If you spot a potential bias in unrelated code, mention it - do not fix it without asking.
- Every changed line must trace directly to the request.

## Goal-driven execution
- Transform vague strategy ideas into verifiable success criteria before implementing.
- For multi-step work, state a plan with a verify step for each item before starting.
