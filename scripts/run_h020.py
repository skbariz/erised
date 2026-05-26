"""H020: ThreeAssetMomentum — standalone walk-forward runner.

Run: uv run python scripts/run_h020.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import MovingAverageCrossover, ThreeAssetMomentum
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    gld = load_research("GLD")[["adj_close"]].rename(columns={"adj_close": "gld_close"})
    prices = spy.join(tlt, how="left").join(gld, how="left")
    prices["tlt_close"] = prices["tlt_close"].ffill()
    prices["gld_close"] = prices["gld_close"].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H020: ThreeAssetMomentum\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} — {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")
    print("Note: TLT available from 2002, GLD from 2004. Early folds → cash.")

    wf = walk_forward_multi(prices, lambda _: ThreeAssetMomentum(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print("\nCanonical (lookback=12m, skip=1m):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    print("\nSensitivity — lookback_months:")
    for lm in [3, 6, 9, 12]:
        w = walk_forward_multi(
            prices, lambda _, l=lm: ThreeAssetMomentum(lookback_months=l), folds=folds
        )
        lo_v, hi_v = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  {lm:>3}m  Sharpe {w.combined_report.sharpe:.3f}  CI [{lo_v:.3f}, {hi_v:.3f}]  MaxDD {w.combined_report.max_drawdown:.2%}"
        )

    # Restated on GLD-available period only (folds 3+)
    print("\nRestated Sharpe (folds 3–11, GLD data available):")
    active_rets = [sharpe(r.net_returns) for r in wf.fold_results[3:]]
    import pandas as pd

    from quant.metrics.performance import sharpe as sh_fn

    combined_active = pd.concat([r.net_returns for r in wf.fold_results[3:]]).sort_index()
    lo_a, hi_a = bootstrap_sharpe_ci(combined_active, n_boot=5000)
    print(f"  Sharpe {sh_fn(combined_active):.3f}  CI [{lo_a:.3f}, {hi_a:.3f}]")

    print("\nPer-fold Sharpe:")
    for i, fold_result in enumerate(wf.fold_results):
        print(
            f"  Fold {i:>2}: {sharpe(fold_result.net_returns):+.3f}  ({folds[i].test_start.date()} — {folds[i].test_end.date()})"
        )

    folds_spy = make_folds(spy, train_years=5, test_years=2)
    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_spy)
    corr = wf.combined_returns.corr(wf_h001.combined_returns.reindex(wf.combined_returns.index))
    print(f"\nH001 correlation: {corr:+.4f}")
    decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
