"""H022: DefensiveEquityHedge — standalone walk-forward runner.

Run: uv run python scripts/run_h022.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies.base import DefensiveEquityHedge
from erised.validation.walk_forward import make_folds, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    prices = spy.join(tlt, how="left")
    prices["tlt_close"] = prices["tlt_close"].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H022: DefensiveEquityHedge (canonical fast=50, slow=200)\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} — {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward_multi(prices, lambda _: DefensiveEquityHedge(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print("\nCanonical (fast=50, slow=200):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    print("\nSensitivity — MA parameters (fast/slow):")
    for fast, slow in [(20, 100), (50, 200), (50, 150)]:
        w = walk_forward_multi(
            prices,
            lambda _, f=fast, s=slow: DefensiveEquityHedge(fast=f, slow=s),
            folds=folds,
        )
        lo_v, hi_v = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  MA({fast}/{slow})  Sharpe {w.combined_report.sharpe:.3f}"
            f"  CI [{lo_v:.3f}, {hi_v:.3f}]"
            f"  MaxDD {w.combined_report.max_drawdown:.2%}"
        )

    print("\nPer-fold Sharpe:")
    for i, fold_result in enumerate(wf.fold_results):
        print(
            f"  Fold {i:>2}: {sharpe(fold_result.net_returns):+.3f}"
            f"  ({folds[i].test_start.date()} — {folds[i].test_end.date()})"
        )

    print(
        f"\nSPY MaxDD: -55.92%  →  H022 MaxDD: {rep.max_drawdown:.2%}  (improvement: {-55.92 - rep.max_drawdown * 100:.1f}pp)"
    )
    decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
