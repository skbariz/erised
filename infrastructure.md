# Infrastructure Reference

Complete guide to every tool, module, and script in this research platform.

---

## Project layout

```
quant/
├── src/erised/              # All importable library code (src layout)
│   ├── strategies/
│   │   ├── base.py         # Strategy ABC + all hypothesis classes
│   │   └── __init__.py     # Public exports
│   ├── backtest/
│   │   └── engine.py       # run_backtest(), BacktestResult
│   ├── metrics/
│   │   └── performance.py  # sharpe(), cagr(), bootstrap_sharpe_ci(), report()
│   ├── validation/
│   │   └── walk_forward.py # make_folds(), walk_forward(), WalkForwardResult
│   ├── data/
│   │   ├── schema.py       # validate_bars()
│   │   ├── ingest.py       # Raw download helpers
│   │   └── loader.py       # load_research(), load_holdout()
│   └── execution/
│       └── paper_trader.py # Paper trading scaffold (future)
├── scripts/                # Standalone runners (not importable)
├── tests/                  # pytest suite
├── data/                   # Parquet files (gitignored)
├── results/                # Per-hypothesis stats.md + SUMMARY.md
└── research_log/           # Narrative write-ups per hypothesis
```

---

## Package manager: uv

All Python commands are run through `uv` to ensure the correct virtualenv.

```bash
uv run python scripts/run_batch1.py   # run a script
uv run pytest                          # run tests
uv run python -c "import erised"        # quick import check
```

`uv` reads `pyproject.toml` for dependencies and manages `.venv` automatically. Never activate the venv manually or use `python` directly — always prefix with `uv run`.

**Key dependencies:** pandas 2.2+, numpy 1.26+, scipy 1.13+, yfinance 0.2.40+, pyarrow 15+, pytest 8+.

---

## Data pipeline

### Files on disk (gitignored, always present in working tree)

| File | Contents | Period |
|------|----------|--------|
| `data/SPY.parquet` | Full SPY OHLCV + adj_close | 1993–present |
| `data/SPY_research.parquet` | Research slice only | 1993–2022 |
| `data/SPY_holdout.parquet` | Sealed holdout — DO NOT TOUCH | 2023+ |
| `data/VIX.parquet` | Full VIX series | 1993–present |
| `data/VIX_research.parquet` | Research slice only | 1993–2022 |
| `data/HOLDOUT_HASH.txt` | SHA-256 of holdout file (committed to git) | — |

### Loading data in code

```python
from erised.data.loader import load_research, load_holdout

# Always use this for strategy development
prices = load_research("SPY")   # returns DataFrame with adj_close, open, high, low, close, volume

# Sealed holdout — only for final model evaluation
prices = load_holdout("SPY", _i_understand_this_is_final_evaluation=True)
```

**Column schema:** `open`, `high`, `low`, `close`, `adj_close`, `volume`. Index is `DatetimeIndex` (business days only).

`load_research()` is safe to call at any time. `load_holdout()` requires the explicit flag by design — accidental calls will raise `RuntimeError` and fail loudly.

### scripts/fetch_data.py

Downloads SPY and VIX from Yahoo Finance via `yfinance`. Run when data needs refreshing or when adding a new ticker.

```bash
uv run python scripts/fetch_data.py
```

To add a new ticker (e.g. TLT for Batch 3), extend the tickers list in this script.

### scripts/seal_holdout.py

Splits the full series at the holdout boundary, writes `*_research.parquet` and `*_holdout.parquet`, and records the holdout hash. Run once per ticker when first adding data. Never re-run for SPY — the holdout is already sealed.

---

## Strategy interface

### `erised.strategies.base.Strategy` (ABC)

Every trading algorithm inherits from `Strategy` and overrides one method:

```python
class Strategy(ABC):
    name: str = "unnamed"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        # Return a Series of target positions in [-1, 1], indexed by date.
        # Position at time t means: "at the close of t, set this target."
        # The backtester shifts by 1 bar — positions are applied the next day.
        ...
```

**Position semantics:**
- `+1.0` = fully long
- `0.0` = flat / cash
- `−1.0` = fully short
- Values between −1 and +1 = scaled partial exposure

**No-lookahead discipline:** All rolling statistics must use `.shift(1)` to exclude today's data from the signal. The engine enforces this structurally by shifting positions one bar, but the internal signal logic is the strategy author's responsibility. `assert_no_lookahead()` is the automated check.

### Strategy classes (H001–H050)

**SPY-only strategies** use `Strategy` base class → `generate_positions()` → `walk_forward()`.

**Multi-asset strategies** use `MultiAssetStrategy` base class → `generate_weights()` → `walk_forward_multi()`. These require additional columns in the prices DataFrame: `tlt_close`, `gld_close`, and/or `vix_close`.

| Class | Hypothesis | Family | Parameters | Data |
|-------|-----------|--------|------------|------|
| `BuyAndHold` | Benchmark | — | none | SPY |
| `MovingAverageCrossover` | H001 | Trend | `fast=50, slow=200, allow_short=False` | SPY |
| `MeanReversionStrategy` | H002 | Mean rev | `lookback=20, threshold=2.0, hold_days=5` | SPY |
| `ShortHoldMeanReversion` | H003 | Mean rev | `lookback=20, threshold=2.0, hold_days=2` | SPY |
| `RSIReversionStrategy` | H004 | Mean rev | `period=14, entry_rsi=30, exit_rsi=50, hold_days=5` | SPY |
| `ConsecutiveDownDays` | H005 | Mean rev | `n_down=3, hold_days=3` | SPY |
| `VIXFilteredReversionStrategy` | H006 | Mean rev | `lookback=20, threshold=2.0, hold_days=2, vix_threshold=20` | SPY+VIX |
| `TimeSeriesMomentum` | H007 | Momentum | `long_months=12, skip_months=1` | SPY |
| `AbsoluteMomentumStrategy` | H008 | Momentum | `lookback_months=12` | SPY |
| `MACDCrossover` | H009 | Trend | `fast=12, slow=26, signal=9` | SPY |
| `FastMACrossover` | H010 | Trend | `fast=10, slow=50` | SPY |
| `ADXFilteredMACrossover` | H011 | Trend | `fast=50, slow=200, adx_period=14, adx_threshold=25` | SPY |
| `VolatilityRegime` | H012 | Vol regime | `vol_lookback=20, regime_lookback=252` | SPY |
| `LongShortMACrossover` | H013 | Trend L/S | `fast=50, slow=200` | SPY |
| `LongShortTSMomentum` | H014 | Momentum L/S | `long_months=12, skip_months=1` | SPY |
| `DualMALongShort` | H015 | Trend L/S | `fast=20, slow=100` | SPY |
| `VolScaledTrend` | H016 | Risk mgmt | `vol_target=0.10, vol_lookback=20, fast=50, slow=200` | SPY |
| `TrendVIXGate` | H017 | Hybrid | `fast=50, slow=200, vix_threshold=40` | SPY+VIX |
| `SPYTLTDualMomentum` | H018 | Multi-asset | `lookback_months=12` | SPY+TLT ★ |
| `SPYGLDRotation` | H019 | Multi-asset | `lookback_months=12` | SPY+GLD ★ |
| `ThreeAssetMomentum` | H020 | Multi-asset | `lookback_months=12` | SPY+TLT+GLD ★ |
| `RiskParity` | H021 | Risk parity | `vol_lookback=60, rebal_threshold=0.05` | SPY+TLT ★ |
| `DefensiveEquityHedge` | H022 | Multi-asset | `fast=50, slow=200, tlt_alloc=0.40` | SPY+TLT ★ |
| `SellInMay` | H023 | Calendar | none | SPY |
| `TurnOfMonth` | H024 | Calendar | `n_days=3` | SPY |
| `JanuaryBarometer` | H025 | Calendar | none | SPY |
| `QuarterEndDrift` | H026 | Calendar | `n_days=5` | SPY |
| `SeasonalityTrendCombo` | H027 | Calendar hybrid | `fast=50, slow=200` | SPY |
| `VIXSpikeEntry` | H028 | VIX contrarian | `spike_threshold=30, hold_days=20` | SPY+VIX |
| `VIXRegimeFilter` | H029 | VIX regime | `vix_threshold=20` | SPY+VIX |
| `VIXTrendSignal` | H030 | VIX L/S | `ma_period=20` | SPY+VIX |
| `RealizedVsImpliedVol` | H031 | VIX regime | `rv_lookback=20` | SPY+VIX |
| `VIXMeanReversion` | H032 | VIX contrarian | `ma_period=60, threshold_sigma=2.0` | SPY+VIX |
| `BondTrendEquityTimer` | H033 | Cross-asset | `ma_period=200` | SPY+TLT |
| `SPYTLTCorrelationRegime` | H034 | Cross-asset | `corr_lookback=60` | SPY+TLT |
| `GoldFearGauge` | H035 | Cross-asset | `ma_period=50` | SPY+GLD |
| `EquityVsSafeHavenMomentum` | H036 | Cross-asset | `lookback_months=3` | SPY+TLT+GLD |
| `DualSafeHavenConfirmation` | H037 | Cross-asset | `ma_period=200` | SPY+TLT+GLD |
| `LongTermReversal` | H038 | Contrarian | `lookback_years=5, pct_threshold=33` | SPY |
| `DeepDrawdownCrisisBuyer` | H039 | Contrarian | `entry_dd=-0.20, exit_dd=-0.05` | SPY |
| `ShortTermWeeklyReversal` | H040 | Mean rev | `hold_days=5, pct_threshold=10` | SPY |
| `PostCrashRecovery` | H041 | Contrarian | `vix_threshold=40, delay_days=30, hold_days=60` | SPY+VIX |
| `VolOfVolCalm` | H042 | Vol signal | `vov_window=20, pct_threshold=33` | SPY+VIX |
| `SellInMayBondConfirm` | H043 | Calendar hybrid | `tlt_ma_period=200` | SPY+TLT |
| `JanuaryBarometerGoldFilter` | H044 | Calendar hybrid | `gld_ma_period=50` | SPY+GLD |
| `TurnOfMonthLowVol` | H045 | Calendar hybrid | `n_days=3, vix_threshold=25` | SPY+VIX |
| `HalloweenMomentumConfirm` | H046 | Calendar hybrid | `momentum_months=6` | SPY |
| `PreEarningsDrift` | H047 | Calendar | `n_days=5` | SPY |

★ Multi-asset strategy — uses `walk_forward_multi()` not `walk_forward()`.

**Ensemble runners** (H048–H050 have no strategy class; they combine component OOS return streams directly):

| Script | Hypothesis | Method |
|--------|-----------|--------|
| `scripts/run_h048.py` | H048 EqualWeightEnsemble | Mean of all 16 kept component returns |
| `scripts/run_h049.py` | H049 MinCorrEnsemble | Greedy max-subset with all pairwise OOS corr < 0.50 |
| `scripts/run_h050.py` | H050 SharpeWeightedEnsemble | Sharpe-proportional weights (clipped at 0) |

---

## Backtest engine

### `erised.backtest.engine.run_backtest()`

```python
from erised.backtest.engine import run_backtest

result = run_backtest(
    prices,           # DataFrame with adj_close
    positions,        # Series of target positions in [-1, 1]
    name="my_strat",
    commission_bps=0.5,   # default
    slippage_bps=1.0,     # default
)
```

**Cost model:** Total cost = `(commission_bps + slippage_bps) / 10000` × `|Δposition|` per period. At 1.5 bps per unit of turnover this is conservative for SPY but appropriate for a small operation.

**`BacktestResult` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `net_returns` | `pd.Series` | Daily returns after all costs — primary metric series |
| `gross_returns` | `pd.Series` | Returns before costs |
| `costs` | `pd.Series` | Cost drag per period |
| `positions` | `pd.Series` | Target positions as generated (not shifted) |
| `applied_positions` | `pd.Series` | Positions actually held (shifted by 1 bar) |
| `equity` | `pd.Series` | Cumulative equity curve, starts at 1.0 |
| `bench_returns` | `pd.Series` | Buy-and-hold benchmark for the same period |
| `bench_equity` | `pd.Series` | Benchmark equity curve |

### `assert_no_lookahead()`

```python
from erised.backtest.engine import assert_no_lookahead

assert_no_lookahead(strategy.generate_positions, prices, n_checks=20)
```

Picks 20 random dates, truncates the price series at each, and verifies that the position at that date is identical whether future data is present or not. Raises `AssertionError` if lookahead is detected. Called in `tests/test_backtest.py` for every strategy.

---

## Walk-forward validation

### `erised.validation.walk_forward.make_folds()`

```python
from erised.validation.walk_forward import make_folds

folds = make_folds(prices, train_years=5, test_years=2)
# Returns list of 12 Fold objects for 1993–2022 SPY research data
```

Default configuration: **5-year train / 2-year test / 12 non-overlapping folds**. Each `Fold` has `train_start`, `train_end`, `test_start`, `test_end` timestamps.

### `walk_forward()`

```python
from erised.validation.walk_forward import walk_forward

wf = walk_forward(
    prices,
    strategy_factory=lambda _: MovingAverageCrossover(50, 200),
    folds=folds,
    commission_bps=0.5,
    slippage_bps=1.0,
)
```

`strategy_factory` receives the train DataFrame for each fold; for parameter-free strategies, ignore it. Returns a `WalkForwardResult`.

**Use `walk_forward()` for all `Strategy` subclasses** (those that implement `generate_positions()`).

### `walk_forward_multi()`

```python
from erised.validation.walk_forward import walk_forward_multi

wf = walk_forward_multi(
    prices,
    strategy_factory=lambda _: RiskParity(),
    folds=folds,
)
```

**Use `walk_forward_multi()` for all `MultiAssetStrategy` subclasses** (H018, H019, H021, H022 — those that implement `generate_weights()`). Calling `walk_forward()` on a `MultiAssetStrategy` will raise `AttributeError` because `generate_positions()` does not exist on those classes.

**Dispatch pattern (used in all ensemble runners):**

```python
from erised.strategies.base import MultiAssetStrategy

def run_wf(prices, strat, folds):
    if isinstance(strat, MultiAssetStrategy):
        return walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
    return walk_forward(prices, lambda _, s=strat: s, folds=folds)
```

**`WalkForwardResult` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `combined_returns` | `pd.Series` | All OOS net returns concatenated |
| `combined_positions` | `pd.Series` | All OOS positions concatenated |
| `combined_report` | `PerformanceReport` | Aggregate metrics across all folds |
| `fold_results` | `list[BacktestResult]` | Per-fold backtest results |
| `folds` | `list[Fold]` | Fold definitions |

---

## Performance metrics

### `erised.metrics.performance`

```python
from erised.metrics.performance import sharpe, cagr, max_drawdown, bootstrap_sharpe_ci, report

sh = sharpe(returns)                          # annualized Sharpe (rf=0)
c = cagr(returns)                             # compound annual growth rate
dd = max_drawdown(returns)                    # most negative drawdown (e.g. -0.30)
lo, hi = bootstrap_sharpe_ci(returns, n_boot=5000)   # 95% CI on Sharpe
rep = report(returns, positions=positions)    # full PerformanceReport
```

**`PerformanceReport` fields:** `sharpe`, `cagr`, `annualized_vol`, `sortino`, `max_drawdown`, `calmar`, `hit_rate`, `avg_win`, `avg_loss`, `turnover`, `n_trades`, `n_periods`, `total_return`.

**Bootstrap CI:** 5,000 i.i.d. resamples. The CI is the primary kill/keep criterion — if the 95% CI lower bound is ≤ 0, the strategy is killed. This guards against noise masquerading as signal.

---

## Test suite

```bash
uv run pytest          # all tests
uv run pytest -v       # verbose
uv run pytest -k H007  # filter by keyword
```

**`tests/test_backtest.py`** — per-strategy tests:
- No-lookahead check via `assert_no_lookahead()`
- Structural behavioral tests (e.g. MA crossover is flat before enough history, positions are in [−1, 1])

**`tests/test_data.py`** — data pipeline tests: schema validation, holdout boundary enforcement.

**`tests/test_metrics.py`** — metric correctness: known-input checks for Sharpe, max drawdown, CAGR.

---

## Batch runner scripts

### scripts/run_hypothesis_002.py

Standalone runner for H002. Runs the walk-forward, prints metrics, and prints sensitivity grid. Used as a template for one-off hypothesis evaluation.

### Batch runners

| Batch | Script | Strategies |
|-------|--------|------------|
| H002 | `scripts/run_hypothesis_002.py` | Mean reversion SPY |
| H003–H012 | `scripts/run_batch1.py` | Trend/momentum/vol batch |
| H013–H017 | `scripts/run_batch2.py` | Long/short structures |
| H018–H022 | `scripts/run_batch3.py` | Multi-asset rotation |
| H023–H027 | `scripts/run_batch4.py` | Seasonality & calendar |
| H028–H032 | `scripts/run_batch5.py` | VIX-based strategies |
| H033–H037 | `scripts/run_batch6.py` | Cross-asset macro timing |
| H038–H042 | `scripts/run_batch7.py` | Contrarian & mean reversion |
| H043–H047 | `scripts/run_batch8.py` | Calendar refinements + cross-asset hybrids |
| H048–H050 | `scripts/run_batch9.py` | Ensemble strategies |

Individual runners (`scripts/run_h<NNN>.py`) exist for each hypothesis and call `write_summary()` at the end to write `results/<NNN>-<name>/summary.json`.

---

## Results structure

```
results/
├── SUMMARY.md                          # Master leaderboard — updated after every hypothesis
├── 001-ma-50-200-baseline/
│   └── stats.md                        # Full OOS metrics, CI, per-fold Sharpe, decision
├── 002-mean-reversion-spy/
│   └── stats.md
...
```

Every `stats.md` contains: parameters, walk-forward config, OOS metrics table, IS reference metrics, per-fold Sharpe breakdown, correlation with H001, and the kill/keep decision with reasoning.

---

## Research log

```
research_log/
├── README.md               # Index of all hypotheses
├── _TEMPLATE.md            # Starting template for new hypotheses
├── 001-ma-50-200-baseline.md
├── 002-mean-reversion-spy.md
...
```

Narrative write-up for each hypothesis: the idea, why it might work, what the result showed, and what it implies for future research. Kept even for killed strategies — the graveyard is the point.

---

## GitHub workflow

**4-commit + 1-issue pattern per hypothesis:**

```
1. gh issue create → issue #N

2. git commit "H0NN: open — <ClassName> hypothesis (issue #N)"
   Files: results/0NN-slug/.gitkeep

3. git commit "H0NN: implement <ClassName>"
   Files: src/erised/strategies/base.py, src/erised/strategies/__init__.py

4. git commit "H0NN: no-lookahead + structural tests; add to run script"
   Files: tests/test_backtest.py, scripts/run_batchN.py

5. git commit "H0NN: results — OOS Sharpe X.XX, KILLED/KEPT (closes #N)"
   Files: results/0NN-slug/stats.md, results/SUMMARY.md, research_log/0NN-slug.md
   Then: gh issue close #N
```

---

## Kill/keep criteria

A strategy is **kept** if all three conditions hold:
1. OOS Sharpe > SPY benchmark (0.51)
2. 95% bootstrap CI lower bound > 0
3. Max drawdown < SPY (−55.92%)

A kept strategy adds ensemble value only if its **return correlation with H001 < 0.30**. Strategies that pass criteria 1–3 but correlate > 0.30 with H001 are kept as standalone signals but do not improve the ensemble.

**Finding from H001–H050:** No strategy achieved H001 corr < 0.30. All 50 strategies had H001 corr > 0.10. The lowest H001 corr among kept strategies was H043 at +0.334. The ensemble (H049) achieved decorrelation not by finding independently low-corr strategies, but by using greedy min-corr subset selection across 16 kept candidates.

---

## Ensemble construction (H048–H050 findings)

The min-correlation greedy algorithm (H049) selects the maximum subset of kept strategies where all pairwise OOS correlations are below a threshold:

```python
def select_min_corr_subset(returns_df, threshold=0.50):
    sharpes = {col: sharpe(returns_df[col].dropna()) for col in returns_df.columns}
    ranked = sorted(sharpes.keys(), key=lambda x: sharpes[x], reverse=True)
    selected = [ranked[0]]
    for candidate in ranked[1:]:
        can_add = True
        for sel in selected:
            common = returns_df[[candidate, sel]].dropna()
            if len(common) < 50:
                continue
            c = common[candidate].corr(common[sel])
            if c > threshold:
                can_add = False
                break
        if can_add:
            selected.append(candidate)
    return selected
```

**Target ensemble (H049):** H021 + H046 + H019 + H035 at threshold 0.50.

| Component | Strategy | OOS Sharpe | H001 corr |
|-----------|----------|-----------|-----------|
| H021 | RiskParity | 1.029 | +0.407 |
| H046 | HalloweenMomentumConfirm | 0.707 | +0.579 |
| H019 | SPYGLDRotation | 0.630 | +0.788 |
| H035 | GoldFearGauge | 0.570 | +0.384 |

Ensemble result: Sharpe 0.959, CI [0.543, 1.357], MaxDD −18.26%, H001 corr +0.697.

**Key insight:** Sharpe-weighting (H050: 0.820) adds negligible value over equal-weight (H048: 0.800). Variance reduction through diversification is the dominant effect — not quality-weighting.

---

## Benchmark reference

SPY buy-and-hold (same OOS windows, 1993–2022 research period):

| Metric | Value |
|--------|-------|
| Sharpe | 0.51 |
| CAGR | 8.35% |
| Max DD | −55.92% |
| Ann vol | 19.66% |
