"""Backtest a 50/200-day moving-average crossover with walk-forward validation.

Runs offline on synthetic prices, so no data download is needed:

    uv run python examples/01_quickstart.py

To use real data instead, run ``erised data fetch`` and ``erised data seal`` once, then
replace ``synthetic_bars()`` with ``erised.data.load_research("SPY")``.
"""

from __future__ import annotations

from erised.backtest.engine import run_backtest
from erised.data.synthetic import synthetic_bars
from erised.metrics.performance import bootstrap_sharpe_ci, report
from erised.strategies import MovingAverageCrossover
from erised.validation.walk_forward import make_folds, walk_forward


def main() -> None:
    prices = synthetic_bars(start="1993-01-29", end="2022-12-30")

    strategy = MovingAverageCrossover(fast=50, slow=200)
    full = run_backtest(prices, strategy.generate_positions(prices), name=strategy.name)
    print("Full-sample backtest (in-sample, for illustration only)")
    print(report(full.net_returns, positions=full.positions))

    folds = make_folds(prices, train_years=5, test_years=2)
    wf = walk_forward(prices, lambda _train: MovingAverageCrossover(50, 200), folds=folds)
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns)

    print(f"\nWalk-forward: {len(folds)} folds, 5y train / 2y test")
    for fold, result in zip(wf.folds, wf.fold_results, strict=True):
        fold_sharpe = report(result.net_returns).sharpe
        print(f"  {fold}  Sharpe {fold_sharpe:+.2f}")
    print(f"\nOut-of-sample Sharpe {wf.combined_report.sharpe:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")
    print(f"Out-of-sample max drawdown {wf.combined_report.max_drawdown:.1%}")


if __name__ == "__main__":
    main()
