# Project
Python research repo for S&P 500 trading strategies. Erised trading models.

Walk-forward OOS methodology: research period 1993-2022, holdout sealed.
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

# Current program state (as of H050 — PROGRAM COMPLETE)
- Strategies tested: 50
- Kept (ensemble candidates): H001 H007 H008 H016 H018 H019 H021 H022 H023 H025 H027 H031 H034 H035 H043 H046 (16 individual) + H048 H049 H050 (3 ensemble) = 19 total
- Star strategy: H021 (RiskParity) — Sharpe 1.029, Max DD -18.41%
- Target ensemble: H049 (MinCorrEnsemble) — Sharpe 0.959, CI [0.543, 1.357], Max DD -18.26%
  Components: H021 (RiskParity) + H046 (HalloweenMomentumConfirm) + H019 (SPYGLDRotation) + H035 (GoldFearGauge)
  All pairwise correlations < 0.50; this is the production-candidate portfolio.
- Best Max DD kept: H043 (SellInMayBondConfirm) at -10.11%
- Lowest H001 corr among kept: H043 (+0.334), H035 (+0.384)
- Key finding: No strategy achieved H001 corr < 0.30. Ensemble decorrelation came from min-corr
  subset selection (H049), not finding individually decorrelated strategies.
- Goal trackers: ALL MET — OOS Sharpe >= 1.0 (H021: 1.03), Max DD < SPY (H043: -10.11%), strategies >= 50 (50 tested)

## Ensemble construction findings (critical for future work)
- Equal-weight of 16 strategies (H048) lifts Sharpe from ~0.61 avg to 0.800 — noise reduction dominates
- Sharpe-weighting (H050: 0.820) barely beats equal-weight (H048: 0.800) — not worth the complexity
- Min-corr subset selection (H049: 0.959) beats both — selecting 4 orthogonal strategies outperforms
  weighting 16 correlated ones
- Contrarian strategies (H038-H042) achieved lowest H001 corr (+0.108 to +0.165) but were killed:
  too few trades (N=6 to N=18 over 30 years) → unreliable OOS estimation
- The correlation/activity tradeoff: strategies most decorrelated from H001 are those that trade
  rarely (crisis buyers, long-term reversal) — too infrequently to clear the keep bar

# Strategy family performance summary (H001–H050)

What each family produced and why — essential context for H051+ design.

## Trend-following (H001, H007, H008, H009–H011, H013–H015, H016, H017, H027)
- Long-only works (H001: 0.61, H007: 0.62, H008: 0.62). Long/short universally fails — shorting adds
  whipsaw costs (1.5 bps/turn) without capturing enough bear-market alpha to compensate.
- Vol-scaling (H016) improves Max DD to -26% but not Sharpe. Hybrid calendar+trend (H027) matches H001.
- All trend signals have H001 corr > 0.70 — they're measuring the same underlying signal.
- **Do not pursue more trend variants.** The family is saturated.

## Multi-asset rotation (H018–H022)
- The highest-value family. Bonds and gold provide genuine crisis diversification.
- RiskParity (H021) is the star of the program: Sharpe 1.03, Max DD -18.41%.
- Rotation strategies (H018, H019) kept but still correlated with H001 (~0.69–0.79) — they hold SPY
  most of the time and only rotate in crises.
- TLT available from 2002, GLD from 2004. Early folds (1993–2002) starved — killed H020 (3-asset).
- **High-value area for H051+:** adding more assets (e.g. commodities, EM bonds, real estate).

## Calendar effects (H023–H027, H043–H047)
- Confirmed real anomalies: Sell-in-May, January Barometer, Halloween effect all pass the keep bar.
- H001 corr range: 0.33–0.71. Calendar signals are the lowest-corr family that reliably keeps.
- Adding cross-asset confirmation (H043 TLT filter, H046 momentum filter) improves Max DD and Sharpe.
- H043 (SellInMayBondConfirm) achieves the best Max DD in the program at -10.11%.
- **Diminishing returns reached.** More calendar hybrids likely to be marginal.

## VIX-based (H028–H032)
- Only H031 (RealizedVsImpliedVol) kept. The vol risk premium (realized < implied = safe to be long)
  is the only durable VIX signal. Pure VIX contrarian (spike entry, mean reversion) fails because
  it enters during crash continuation, not after capitulation.
- H030 (VIX trend L/S) proved negative H001 corr (-0.284) is possible but at negative Sharpe.

## Cross-asset macro (H033–H037)
- Bond trend (H034) and gold fear gauge (H035) kept. Both are slow-moving regime indicators.
- H037 (dual safe-haven confirmation) achieved lowest H001 corr of any kept strategy (+0.264) but
  was too restrictive: 5 zero-trade folds → killed.
- H035 is an anchor of the target ensemble (H049) despite high H001 corr — pairwise corr with
  H021 and H046 is low enough to add ensemble value.

## Contrarian / mean reversion (H038–H042)
- ALL killed. Lowest H001 corr in the program (+0.108 to +0.165) but N=6–18 trades over 30 years.
- The fundamental tradeoff: strategies that fire only in crises are most decorrelated but least
  tradeable. No fix exists within this framework — more data or multi-market generalization needed.
- **Do not pursue more contrarian signals** unless you have a mechanism to increase trade frequency.

## Ensemble (H048–H050)
- Min-corr subset (H049) at Sharpe 0.959 beats equal-weight (H048: 0.800) and Sharpe-weighted (H050: 0.820).
- Key insight: 4 orthogonal strategies outperform 16 correlated ones. Variance reduction > quality-weighting.

# Research directions for H051+

Ranked by expected information value (what we don't yet know):

1. **Equity factors (value, quality, low-vol, size)** — SPY-only but signal is fundamentally different
   from price momentum. Academic evidence strong. Expected H001 corr: 0.10–0.40.
2. **Additional asset classes** — commodities index (DBC), real estate (VNQ), EM bonds (EMB) as
   rotation candidates in multi-asset framework. Builds on H021's success.
3. **Macro regime conditioning** — yield curve slope (10y−2y), credit spread (HYG−LQD) as overlay
   signals. Turn off equity exposure when yield curve inverts or spreads spike.
4. **Holdout evaluation** — Before H051, run H049 (the target ensemble) on the sealed 2023+ data
   to get a true out-of-sample read on the program's flagship result.

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
