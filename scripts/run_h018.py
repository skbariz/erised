"""H018: SPYTLTDualMomentum — standalone walk-forward runner.

Run: uv run python scripts/run_h018.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, report, sharpe
from quant.strategies.base import MovingAverageCrossover, SPYTLTDualMomentum
from quant.validation.walk_forward import make_folds, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    prices = spy.join(tlt, how="left")
    prices["tlt_close"] = prices["tlt_close"].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H018: SPYTLTDualMomentum\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} — {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    # H001 baseline for correlation reference
    wf_h001 = walk_forward_multi.__wrapped__ if hasattr(walk_forward_multi, "__wrapped__") else None

    # Canonical
    wf = walk_forward_multi(prices, strategy_factory=lambda _: SPYTLTDualMomentum(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print(f"\nCanonical (lookback=12m, skip=1m):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    # Sensitivity
    print("\nSensitivity — lookback_months:")
    for lm in [3, 6, 9, 12, 18]:
        w = walk_forward_multi(prices, lambda _, l=lm: SPYTLTDualMomentum(lookback_months=l), folds=folds)
        lo_v, hi_v = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(f"  {lm:>3}m  Sharpe {w.combined_report.sharpe:.3f}  CI [{lo_v:.3f}, {hi_v:.3f}]  MaxDD {w.combined_report.max_drawdown:.2%}")

    # Per-fold
    print("\nPer-fold Sharpe:")
    for i, fold_result in enumerate(wf.fold_results):
        sh = sharpe(fold_result.net_returns)
        print(f"  Fold {i:>2}: {sh:+.3f}  ({folds[i].test_start.date()} — {folds[i].test_end.date()})")

    # H001 correlation
    from quant.validation.walk_forward import walk_forward
    spy_only = load_research("SPY")
    folds_spy = make_folds(spy_only, train_years=5, test_years=2)
    wf_h001 = walk_forward(spy_only, lambda _: MovingAverageCrossover(50, 200), folds=folds_spy)
    corr = wf.combined_returns.corr(wf_h001.combined_returns.reindex(wf.combined_returns.index))
    print(f"\nH001 correlation: {corr:+.4f}")
    decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
