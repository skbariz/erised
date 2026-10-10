"""H042: VolOfVolCalm -- standalone walk-forward runner.

Run: uv run python scripts/run_h042.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies.base import MovingAverageCrossover, VolOfVolCalm
from erised.validation.walk_forward import make_folds, walk_forward

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})
    prices = spy.join(vix, how="left")
    prices["vix_close"] = prices["vix_close"].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H042: VolOfVolCalm\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward(prices, lambda _: VolOfVolCalm(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print("\nCanonical (vov_window=20d, percentile=33):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    print("\nSensitivity -- vov_window:")
    for vw in [10, 20, 40, 60]:
        w = walk_forward(prices, lambda _, v=vw: VolOfVolCalm(vov_window=v), folds=folds)
        lo_s, hi_s = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  {vw:>3}d  Sharpe {w.combined_report.sharpe:.3f}"
            f"  CI [{lo_s:.3f}, {hi_s:.3f}]  MaxDD {w.combined_report.max_drawdown:.2%}"
        )

    print("\nSensitivity -- percentile:")
    for pct in [20, 33, 50]:
        w = walk_forward(prices, lambda _, p=pct: VolOfVolCalm(percentile=p), folds=folds)
        print(f"  p{pct:<3}  Sharpe {w.combined_report.sharpe:.3f}")

    print("\nPer-fold Sharpe:")
    folds_spy = make_folds(spy, train_years=5, test_years=2)
    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_spy)
    for i, fold_result in enumerate(wf.fold_results):
        print(
            f"  Fold {i:>2}: {sharpe(fold_result.net_returns):+.3f}"
            f"  ({folds[i].test_start.date()} -- {folds[i].test_end.date()})"
        )

    corr = wf.combined_returns.corr(wf_h001.combined_returns.reindex(wf.combined_returns.index))
    print(f"\nH001 correlation: {corr:+.4f}")
    decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="042",
        name="vol-of-vol-calm",
        strategy_class="VolOfVolCalm",
        family="vol signal",
        sharpe=rep.sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=rep.cagr,
        max_drawdown=rep.max_drawdown,
        n_trades=rep.n_trades,
        h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results],
        decision=decision,
        notes="Long when VIX 20d std dev in bottom tercile (stable fear = calm regime). Second-order VIX signal.",
        canonical_params={"vov_window": 20, "percentile": 33.0},
    )


if __name__ == "__main__":
    main()
