"""H026: QuarterEndDrift — standalone walk-forward runner.

Run: uv run python scripts/run_h026.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import QuarterEndDrift
from quant.validation.walk_forward import make_folds, walk_forward

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    folds = make_folds(spy, train_years=5, test_years=2)

    print(f"H026: QuarterEndDrift (canonical: n_days=5)\n{SEP}")
    print(f"Data: {len(spy)} rows  {spy.index.min().date()} — {spy.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward(spy, lambda _: QuarterEndDrift(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print(f"\nCanonical (n_days=5):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    print("\nSensitivity — n_days (before quarter-end):")
    for nd in [3, 5, 10]:
        w = walk_forward(spy, lambda _, n=nd: QuarterEndDrift(n_days=n), folds=folds)
        lo_v, hi_v = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  n_days={nd:>2}  Sharpe {w.combined_report.sharpe:.3f}"
            f"  CI [{lo_v:.3f}, {hi_v:.3f}]"
            f"  MaxDD {w.combined_report.max_drawdown:.2%}"
            f"  N_trades {w.combined_report.n_trades}"
        )

    print("\nPer-fold Sharpe:")
    for i, fold_result in enumerate(wf.fold_results):
        print(
            f"  Fold {i:>2}: {sharpe(fold_result.net_returns):+.3f}"
            f"  ({folds[i].test_start.date()} — {folds[i].test_end.date()})"
        )

    print(f"\nLowest H001 correlation in the program: +0.176")
    print(f"Lowest MaxDD in the program: {rep.max_drawdown:.2%}")
    decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
