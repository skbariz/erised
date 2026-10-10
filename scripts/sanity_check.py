"""End-to-end sanity check.

Loads research data, runs buy-and-hold and MA crossover strategies through
the backtest engine and walk-forward validator, and prints metrics.

Use it to confirm the data, engine, and validation layers plumb together.
On synthetic data the numbers are not meaningful; the goal is to verify the
pipeline runs and produces sensible outputs.

Run: uv run python scripts/sanity_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from erised.backtest.engine import run_backtest
from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, report
from erised.strategies import BuyAndHold, MovingAverageCrossover
from erised.validation.walk_forward import make_folds, walk_forward


def fmt_report(name: str, prices, rep) -> str:
    out = [f"\n{'=' * 60}", f"  {name}", "=" * 60, str(rep)]
    return "\n".join(out)


def main() -> None:
    prices = load_research("SPY")
    print(f"Loaded {len(prices)} rows: {prices.index.min().date()} -> {prices.index.max().date()}")

    # --- In-sample backtest of two sanity strategies ---
    for strat in [BuyAndHold(), MovingAverageCrossover(50, 200)]:
        positions = strat.generate_positions(prices)
        bt = run_backtest(prices, positions, name=strat.name)
        rep = report(bt.net_returns, positions=bt.positions)
        print(fmt_report(f"IN-SAMPLE: {strat.name}", prices, rep))

    # --- Walk-forward on the MA crossover ---
    print(f"\n{'=' * 60}")
    print("  WALK-FORWARD: MA(50,200)")
    print("=" * 60)
    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"Generated {len(folds)} folds:")
    for f in folds:
        print(f"  {f}")

    wf = walk_forward(
        prices,
        strategy_factory=lambda _train: MovingAverageCrossover(50, 200),
        folds=folds,
    )
    print(f"\nCombined OOS report (across {len(folds)} folds):")
    print(wf.combined_report)

    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=2000)
    print(f"\nBootstrap 95% CI on OOS Sharpe: [{lo:.2f}, {hi:.2f}]")
    print(f"Point estimate: {wf.combined_report.sharpe:.2f}")

    # --- Benchmark comparison ---
    import pandas as pd

    bench_returns = pd.concat([f.bench_returns for f in wf.fold_results]).sort_index()
    bench_rep = report(bench_returns)
    print("\nBenchmark (buy-and-hold) over same OOS periods:")
    print(bench_rep)


if __name__ == "__main__":
    main()
