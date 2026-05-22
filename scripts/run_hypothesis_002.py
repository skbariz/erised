"""Walk-forward evaluation of Hypothesis 002: short-term mean reversion on SPY.

Run: uv run python scripts/run_hypothesis_002.py

Produces:
  - Canonical OOS metrics for MeanReversionStrategy(lookback=20, threshold=2.0, hold_days=5)
  - Bootstrap 95% CI on OOS Sharpe (10,000 resamples)
  - Per-fold Sharpe breakdown for regime analysis
  - 1-D sensitivity tables (one param varied, others at canonical)
  - Stability summary across the full 27-combination parameter grid
  - Pearson correlation with H001 (MA 50/200) over identical OOS folds
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, report, sharpe
from quant.strategies.base import MeanReversionStrategy, MovingAverageCrossover
from quant.validation.walk_forward import make_folds, walk_forward

SEP = "=" * 62
CANONICAL = dict(lookback=20, threshold=2.0, hold_days=5)
PARAM_GRID = list(product([10, 20, 30], [1.5, 2.0, 2.5], [3, 5, 7]))


def section(title: str) -> str:
    return f"\n{SEP}\n  {title}\n{SEP}"


def run_wf(prices, folds, lookback, threshold, hold_days):
    return walk_forward(
        prices,
        strategy_factory=lambda _: MeanReversionStrategy(lookback, threshold, hold_days),
        folds=folds,
    )


def main() -> None:
    prices = load_research("SPY")
    print(f"Research data: {len(prices)} rows  {prices.index.min().date()} to {prices.index.max().date()}")

    # Identical folds to H001 — required for a valid OOS correlation comparison.
    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"\nWalk-forward: {len(folds)} folds (5-year train / 2-year test)")
    for f in folds:
        print(f"  {f}")

    # ------------------------------------------------------------------ #
    # Canonical run                                                         #
    # ------------------------------------------------------------------ #
    print(section(
        f"H002 CANONICAL: MeanReversionStrategy"
        f"(lookback={CANONICAL['lookback']}, threshold={CANONICAL['threshold']}, hold_days={CANONICAL['hold_days']})"
    ))
    wf_canon = run_wf(prices, folds, **CANONICAL)
    rep = wf_canon.combined_report
    print(f"\nCombined OOS report ({len(folds)} folds, {rep.n_periods} periods):")
    print(rep)

    lo, hi = bootstrap_sharpe_ci(wf_canon.combined_returns, n_boot=10_000)
    print(f"\nBootstrap 95% CI on OOS Sharpe: [{lo:.2f}, {hi:.2f}]")

    bench_returns = pd.concat([r.bench_returns for r in wf_canon.fold_results]).sort_index()
    bench_rep = report(bench_returns)
    print(f"\nSPY benchmark over same OOS windows:")
    print(f"  Sharpe {bench_rep.sharpe:.2f}  CAGR {bench_rep.cagr:.2%}  Max DD {bench_rep.max_drawdown:.2%}")

    # ------------------------------------------------------------------ #
    # Per-fold Sharpe (regime analysis)                                    #
    # ------------------------------------------------------------------ #
    print(section("PER-FOLD OOS SHARPE  (regime / decay check)"))
    print(f"  {'Fold':<5} {'Test window':<25} {'H002':>8} {'SPY':>8}")
    print(f"  {'-'*5} {'-'*25} {'-'*8} {'-'*8}")
    for fold, result in zip(wf_canon.folds, wf_canon.fold_results):
        h002_sh = sharpe(result.net_returns)
        spy_sh = sharpe(result.bench_returns)
        window = f"{fold.test_start.date()} to {fold.test_end.date()}"
        print(f"  {fold.fold_idx:<5} {window:<25} {h002_sh:>8.2f} {spy_sh:>8.2f}")

    # ------------------------------------------------------------------ #
    # Pre-compute all 27 grid points                                        #
    # ------------------------------------------------------------------ #
    print(section("RUNNING 27-COMBINATION PARAMETER GRID  (please wait)"))
    grid_results: dict[tuple, float] = {}
    for lb, thr, hd in PARAM_GRID:
        wf = run_wf(prices, folds, lb, thr, hd)
        grid_results[(lb, thr, hd)] = wf.combined_report.sharpe
        print(f"  lookback={lb:2d}  threshold={thr:.1f}  hold_days={hd}  Sharpe {wf.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ #
    # 1-D sensitivity tables                                               #
    # ------------------------------------------------------------------ #
    print(section("1-D SENSITIVITY  (one param varied, others at canonical)"))

    for param, values, label in [
        ("lookback",  [10, 20, 30],   "threshold=2.0, hold_days=5"),
        ("threshold", [1.5, 2.0, 2.5], "lookback=20,  hold_days=5"),
        ("hold_days", [3, 5, 7],      "lookback=20,  threshold=2.0"),
    ]:
        print(f"\n  Varying {param}  ({label})")
        print(f"  {'Value':>9}  {'OOS Sharpe':>12}")
        for v in values:
            key = {
                "lookback":  (v, CANONICAL["threshold"], CANONICAL["hold_days"]),
                "threshold": (CANONICAL["lookback"], v, CANONICAL["hold_days"]),
                "hold_days": (CANONICAL["lookback"], CANONICAL["threshold"], v),
            }[param]
            sh = grid_results[key]
            marker = " <-- canonical" if v == CANONICAL[param] else ""
            print(f"  {v:>9}  {sh:>12.2f}{marker}")

    # ------------------------------------------------------------------ #
    # Stability summary                                                     #
    # ------------------------------------------------------------------ #
    print(section("STABILITY SUMMARY  (all 27 combinations)"))
    arr = np.array(list(grid_results.values()))
    print(f"  N combinations:       {len(arr)}")
    print(f"  Median OOS Sharpe:    {np.median(arr):.2f}")
    print(f"  IQR:                  [{np.percentile(arr, 25):.2f}, {np.percentile(arr, 75):.2f}]")
    print(f"  Count Sharpe > 0.5:   {(arr > 0.5).sum()} / {len(arr)}")
    print(f"  Min / Max:            {arr.min():.2f} / {arr.max():.2f}")

    # ------------------------------------------------------------------ #
    # Correlation with H001 over identical OOS folds                       #
    # ------------------------------------------------------------------ #
    print(section("CORRELATION WITH H001  (MA 50/200, same folds)"))
    wf_h001 = walk_forward(
        prices,
        strategy_factory=lambda _: MovingAverageCrossover(50, 200),
        folds=folds,
    )
    r_corr = wf_canon.combined_returns.corr(wf_h001.combined_returns)
    p_corr = wf_canon.combined_positions.corr(wf_h001.combined_positions)
    print(f"  Pearson correlation (net returns):   {r_corr:+.3f}")
    print(f"  Pearson correlation (positions):     {p_corr:+.3f}")
    verdict = "PASS (< 0.30)" if abs(r_corr) < 0.30 else "FAIL (>= 0.30)"
    print(f"  Ensemble diversification check:      {verdict}")
    print(f"\n  H001 OOS Sharpe (same windows):      {wf_h001.combined_report.sharpe:.2f}")
    print(f"  H002 OOS Sharpe (canonical):         {rep.sharpe:.2f}")


if __name__ == "__main__":
    main()
