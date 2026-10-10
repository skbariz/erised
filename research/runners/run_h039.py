"""H039: DeepDrawdownCrisisBuyer -- standalone walk-forward runner.

Run: uv run python scripts/run_h039.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies import DeepDrawdownCrisisBuyer, MovingAverageCrossover
from erised.validation.walk_forward import make_folds, walk_forward

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    folds = make_folds(spy, train_years=5, test_years=2)
    print(f"H039: DeepDrawdownCrisisBuyer\n{SEP}")
    print(f"Data: {len(spy)} rows  {spy.index.min().date()} -- {spy.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward(spy, lambda _: DeepDrawdownCrisisBuyer(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print("\nCanonical (entry_dd=-20%, exit_dd=-5%, lookback=252d):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    print("\nSensitivity -- entry_dd:")
    for ed in [-0.10, -0.15, -0.20, -0.25, -0.30]:
        w = walk_forward(spy, lambda _, e=ed: DeepDrawdownCrisisBuyer(entry_dd=e), folds=folds)
        lo_s, hi_s = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  entry={ed:.0%}  Sharpe {w.combined_report.sharpe:.3f}"
            f"  CI [{lo_s:.3f}, {hi_s:.3f}]  MaxDD {w.combined_report.max_drawdown:.2%}"
        )

    print("\nSensitivity -- exit_dd:")
    for xd in [-0.10, -0.05, 0.00]:
        w = walk_forward(spy, lambda _, x=xd: DeepDrawdownCrisisBuyer(exit_dd=x), folds=folds)
        print(f"  exit={xd:.0%}  Sharpe {w.combined_report.sharpe:.3f}")

    print("\nPer-fold Sharpe:")
    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds)
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
        hypothesis_id="039",
        name="deep-drawdown-crisis-buyer",
        strategy_class="DeepDrawdownCrisisBuyer",
        family="contrarian",
        sharpe=rep.sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=rep.cagr,
        max_drawdown=rep.max_drawdown,
        n_trades=rep.n_trades,
        h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results],
        decision=decision,
        notes="Long when drawdown from 1yr peak exceeds 20%; exit when recovered to -5%.",
        canonical_params={"entry_dd": -0.20, "exit_dd": -0.05, "lookback": 252},
    )


if __name__ == "__main__":
    main()
