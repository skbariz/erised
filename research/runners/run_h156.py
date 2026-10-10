"""H156: JumpPenalizedMomentum — trimmed-return momentum (exclude top/bottom 5 days).

Run: uv run python scripts/run_h156.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from erised.data.loader import load_research
from erised.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies.base import MovingAverageCrossover
from erised.strategies.cross_sectional import JumpPenalizedMomentum, MomentumLongOnly
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional

SEP = "=" * 64


def main() -> None:
    print(f"H156: JumpPenalizedMomentum (trimmed return, top/bottom 5 days removed)\n{SEP}")

    membership = load_sp500_members()
    tickers = get_all_historical_tickers()
    print(f"Loading prices for {len(tickers)} historical tickers...")
    panel = load_stocks_panel(tickers)

    common_idx = panel.index.intersection(membership.index)
    panel = panel.loc[common_idx]
    membership = membership.loc[common_idx]

    spy = load_research("SPY")
    spy_rets = spy["adj_close"].pct_change().fillna(0.0)

    folds = make_folds(panel, train_years=5, test_years=2)
    print(f"Data: {panel.shape[0]} days, {panel.shape[1]} tickers")
    print(f"Folds: {len(folds)}")

    wf = walk_forward_cross_sectional(
        panel,
        membership,
        lambda tp, tm: JumpPenalizedMomentum(monthly_rebalance=True),
        folds=folds,
        spy_returns=spy_rets,
    )
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)

    print("\nH156 Canonical (trim_days=5, top_quantile=20%):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}")

    wf_h150 = walk_forward_cross_sectional(
        panel,
        membership,
        lambda tp, tm: MomentumLongOnly(),
        folds=folds,
        spy_returns=spy_rets,
    )
    print(
        f"\nH150 baseline: Sharpe {wf_h150.combined_report.sharpe:.3f}"
        f"  MaxDD {wf_h150.combined_report.max_drawdown:.2%}"
    )

    print("\nSensitivity -- trim_days:")
    for trim in [3, 5, 10]:
        w = walk_forward_cross_sectional(
            panel,
            membership,
            lambda tp, tm, _t=trim: JumpPenalizedMomentum(trim_days=_t),
            folds=folds,
            spy_returns=spy_rets,
        )
        lo_t, hi_t = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  trim_days={trim:>2}  Sharpe {w.combined_report.sharpe:.3f}"
            f"  CI [{lo_t:.3f}, {hi_t:.3f}]  MaxDD {w.combined_report.max_drawdown:.2%}"
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

    decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="156",
        name="jump-penalized-momentum",
        strategy_class="JumpPenalizedMomentum",
        family="cross-sectional momentum",
        sharpe=rep.sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=rep.cagr,
        max_drawdown=rep.max_drawdown,
        n_trades=rep.n_trades,
        h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results],
        decision=decision,
        notes="12-1m trimmed return (exclude top/bottom 5 daily returns). Rewards steady-trend stocks.",
        canonical_params={
            "long_days": 252,
            "skip_days": 21,
            "top_quantile": 0.20,
            "trim_days": 5,
            "monthly_rebalance": True,
        },
    )


if __name__ == "__main__":
    main()
