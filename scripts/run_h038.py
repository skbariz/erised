"""H038: LongTermReversal -- standalone walk-forward runner.

Run: uv run python scripts/run_h038.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import LongTermReversal, MovingAverageCrossover
from quant.validation.walk_forward import make_folds, walk_forward

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    folds = make_folds(spy, train_years=5, test_years=2)
    print(f"H038: LongTermReversal\n{SEP}")
    print(f"Data: {len(spy)} rows  {spy.index.min().date()} -- {spy.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward(spy, lambda _: LongTermReversal(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print("\nCanonical (lookback=1260d, percentile=33):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    print("\nSensitivity -- lookback_days:")
    for lb in [504, 756, 1008, 1260, 1512]:
        w = walk_forward(spy, lambda _, lb_=lb: LongTermReversal(lookback_days=lb_), folds=folds)
        lo_s, hi_s = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  {lb:>5}d  Sharpe {w.combined_report.sharpe:.3f}"
            f"  CI [{lo_s:.3f}, {hi_s:.3f}]  MaxDD {w.combined_report.max_drawdown:.2%}"
        )

    print("\nSensitivity -- percentile:")
    for pct in [20, 25, 33, 40]:
        w = walk_forward(spy, lambda _, p=pct: LongTermReversal(percentile=p), folds=folds)
        print(f"  p{pct:<3}  Sharpe {w.combined_report.sharpe:.3f}")

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
        hypothesis_id="038",
        name="long-term-reversal",
        strategy_class="LongTermReversal",
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
        notes="Long when 5yr rolling return in bottom tercile of historical distribution (DeBondt-Thaler 1985).",
        canonical_params={"lookback_days": 1260, "percentile": 33.0},
    )


if __name__ == "__main__":
    main()
