"""H027: SeasonalityTrendCombo — standalone walk-forward runner.

Run: uv run python scripts/run_h027.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import MovingAverageCrossover, SeasonalityTrendCombo, SellInMay
from quant.validation.walk_forward import make_folds, walk_forward

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    folds = make_folds(spy, train_years=5, test_years=2)

    print(f"H027: SeasonalityTrendCombo (MA(50,200) × seasonal gate)\n{SEP}")
    print(f"Data: {len(spy)} rows  {spy.index.min().date()} — {spy.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward(spy, lambda _: SeasonalityTrendCombo(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print(f"\nCanonical (fast=50, slow=200, Nov-Apr gate):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    print("\nSensitivity — MA parameters:")
    for fast, slow in [(20, 100), (50, 200), (50, 150)]:
        w = walk_forward(
            spy,
            lambda _, f=fast, s=slow: SeasonalityTrendCombo(fast=f, slow=s),
            folds=folds,
        )
        lo_v, hi_v = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  MA({fast}/{slow})  Sharpe {w.combined_report.sharpe:.3f}"
            f"  CI [{lo_v:.3f}, {hi_v:.3f}]"
            f"  MaxDD {w.combined_report.max_drawdown:.2%}"
        )

    print("\nComparison with component strategies:")
    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds)
    wf_h023 = walk_forward(spy, lambda _: SellInMay(), folds=folds)
    corr_h001 = wf.combined_returns.corr(wf_h001.combined_returns.reindex(wf.combined_returns.index))
    corr_h023 = wf.combined_returns.corr(wf_h023.combined_returns.reindex(wf.combined_returns.index))
    print(f"  H001 (MA crossover) Sharpe: {wf_h001.combined_report.sharpe:.3f}  H027 corr: {corr_h001:+.3f}")
    print(f"  H023 (SellInMay)    Sharpe: {wf_h023.combined_report.sharpe:.3f}  H027 corr: {corr_h023:+.3f}")
    print(f"  H027 (this)         Sharpe: {rep.sharpe:.3f}")

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
