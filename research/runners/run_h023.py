"""H023: SellInMay — standalone walk-forward runner.

Run: uv run python scripts/run_h023.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies import SellInMay
from erised.validation.walk_forward import make_folds, walk_forward

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    folds = make_folds(spy, train_years=5, test_years=2)

    print(f"H023: SellInMay ★ (canonical: favorable_months=Nov-Apr)\n{SEP}")
    print(f"Data: {len(spy)} rows  {spy.index.min().date()} — {spy.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward(spy, lambda _: SellInMay(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print("\nCanonical (Nov-Apr):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    print("\nSensitivity — favorable window:")
    windows = [
        ((10, 11, 12, 1, 2, 3, 4), "Oct-Apr (7m)"),
        ((11, 12, 1, 2, 3, 4), "Nov-Apr (6m, canonical)"),
        ((11, 12, 1, 2, 3), "Nov-Mar (5m)"),
    ]
    for months, label in windows:
        w = walk_forward(spy, lambda _, m=months: SellInMay(favorable_months=m), folds=folds)
        lo_v, hi_v = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  {label:<25}  Sharpe {w.combined_report.sharpe:.3f}"
            f"  CI [{lo_v:.3f}, {hi_v:.3f}]"
            f"  MaxDD {w.combined_report.max_drawdown:.2%}"
        )

    print("\nPer-fold Sharpe:")
    for i, fold_result in enumerate(wf.fold_results):
        print(
            f"  Fold {i:>2}: {sharpe(fold_result.net_returns):+.3f}"
            f"  ({folds[i].test_start.date()} — {folds[i].test_end.date()})"
        )

    decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"
    print(f"\nDecision: {decision}")


if __name__ == "__main__":
    main()
