"""H060: CarryMomentumSPYTLT -- standalone walk-forward runner.

Run: uv run python scripts/run_h060.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies.base import CarryMomentumSPYTLT, MovingAverageCrossover
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    shy = load_research("SHY")[["adj_close"]].rename(columns={"adj_close": "shy_close"})
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    prices = spy.join(shy, how="left").join(tlt, how="left")
    prices["shy_close"] = prices["shy_close"].ffill()
    prices["tlt_close"] = prices["tlt_close"].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H060: CarryMomentumSPYTLT (canonical mom_days=63, spy_ma=200)\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward_multi(prices, lambda _: CarryMomentumSPYTLT(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print("\nCanonical (mom_days=63, spy_ma=200):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    print("\nSensitivity -- mom_days (spy_ma=200):")
    for md in [21, 42, 63, 126]:
        w = walk_forward_multi(
            prices, lambda _, m=md: CarryMomentumSPYTLT(mom_days=m, spy_ma=200), folds=folds
        )
        lo_v, hi_v = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  mom={md}d  Sharpe {w.combined_report.sharpe:.3f}"
            f"  CI [{lo_v:.3f}, {hi_v:.3f}]  MaxDD {w.combined_report.max_drawdown:.2%}"
        )

    print("\nSensitivity -- spy_ma (mom_days=63):")
    for sm in [50, 100, 150, 200, 252]:
        w = walk_forward_multi(
            prices, lambda _, s=sm: CarryMomentumSPYTLT(mom_days=63, spy_ma=s), folds=folds
        )
        lo_v, hi_v = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  spy_ma={sm}d  Sharpe {w.combined_report.sharpe:.3f}"
            f"  CI [{lo_v:.3f}, {hi_v:.3f}]  MaxDD {w.combined_report.max_drawdown:.2%}"
        )

    print("\nPer-fold Sharpe:")
    for i, fold_result in enumerate(wf.fold_results):
        print(
            f"  Fold {i:>2}: {sharpe(fold_result.net_returns):+.3f}"
            f"  ({folds[i].test_start.date()} -- {folds[i].test_end.date()})"
        )

    folds_spy = make_folds(spy, train_years=5, test_years=2)
    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_spy)
    corr = wf.combined_returns.corr(wf_h001.combined_returns.reindex(wf.combined_returns.index))
    print(f"\nH001 correlation: {corr:+.4f}")
    print(
        f"SPY MaxDD: -55.92%  ->  H060 MaxDD: {rep.max_drawdown:.2%}"
        f"  (improvement: {-55.92 - rep.max_drawdown * 100:.1f}pp)"
    )
    decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="060",
        name="carry-momentum-spy-tlt",
        strategy_class="CarryMomentumSPYTLT",
        family="multi-asset rotation",
        sharpe=rep.sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=rep.cagr,
        max_drawdown=rep.max_drawdown,
        n_trades=rep.n_trades,
        h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results],
        decision=decision,
        notes="H056 + SPY 200d MA gate; flat when curve steep but SPY below trend.",
        canonical_params={"mom_days": 63, "spy_ma": 200},
    )


if __name__ == "__main__":
    main()
