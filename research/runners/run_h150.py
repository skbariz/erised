"""H150: MomentumLongOnly — cross-sectional 12-1 momentum, long-only equal weight.

Run: uv run python scripts/run_h150.py

Requires data/sp500_members.parquet and data/stocks/ populated.
See scripts/fetch_sp500_members.py and scripts/fetch_stocks.py.
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
from erised.strategies.cross_sectional import MomentumLongOnly
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional

SEP = "=" * 64


def main() -> None:
    print(f"H150: MomentumLongOnly (12-1m, long-only, top quintile)\n{SEP}")

    membership = load_sp500_members()
    tickers = get_all_historical_tickers()
    print(f"Loading prices for {len(tickers)} historical tickers...")
    panel = load_stocks_panel(tickers)

    # Align panel and membership to common dates
    common_idx = panel.index.intersection(membership.index)
    panel = panel.loc[common_idx]
    membership = membership.loc[common_idx]

    spy = load_research("SPY")
    spy_rets = spy["adj_close"].pct_change().fillna(0.0)

    folds = make_folds(panel, train_years=5, test_years=2)
    print(
        f"Data: {panel.shape[0]} days, {panel.shape[1]} tickers  "
        f"{panel.index.min().date()} -- {panel.index.max().date()}"
    )
    print(f"Folds: {len(folds)}")

    wf = walk_forward_cross_sectional(
        panel,
        membership,
        lambda tp, tm: MomentumLongOnly(monthly_rebalance=True),
        folds=folds,
        spy_returns=spy_rets,
    )
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)

    print("\nH150 Canonical (12-1m, top 20%, monthly rebalance, 2.5bps/turn):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}")

    # Rebalance frequency comparison
    wf_daily = walk_forward_cross_sectional(
        panel,
        membership,
        lambda tp, tm: MomentumLongOnly(monthly_rebalance=False),
        folds=folds,
        spy_returns=spy_rets,
    )
    print(
        f"\nRebalance comparison:"
        f"\n  Monthly: Sharpe {rep.sharpe:.3f}  MaxDD {rep.max_drawdown:.2%}"
        f"\n  Daily:   Sharpe {wf_daily.combined_report.sharpe:.3f}"
        f"  MaxDD {wf_daily.combined_report.max_drawdown:.2%}"
    )

    print("\nSensitivity -- top_quantile:")
    for q in [0.10, 0.20, 0.30, 0.40]:
        w = walk_forward_cross_sectional(
            panel,
            membership,
            lambda tp, tm, _q=q: MomentumLongOnly(top_quantile=_q),
            folds=folds,
            spy_returns=spy_rets,
        )
        lo_q, hi_q = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  top_quantile={q:.0%}  Sharpe {w.combined_report.sharpe:.3f}"
            f"  CI [{lo_q:.3f}, {hi_q:.3f}]  MaxDD {w.combined_report.max_drawdown:.2%}"
        )

    print("\nSensitivity -- skip_days:")
    for skip in [0, 21, 42]:
        w = walk_forward_cross_sectional(
            panel,
            membership,
            lambda tp, tm, s=skip: MomentumLongOnly(skip_days=s),
            folds=folds,
            spy_returns=spy_rets,
        )
        print(
            f"  skip_days={skip:>2}  Sharpe {w.combined_report.sharpe:.3f}"
            f"  MaxDD {w.combined_report.max_drawdown:.2%}"
        )

    print("\nPer-fold Sharpe:")
    for i, fold_result in enumerate(wf.fold_results):
        print(
            f"  Fold {i:>2}: {sharpe(fold_result.net_returns):+.3f}"
            f"  ({folds[i].test_start.date()} -- {folds[i].test_end.date()})"
        )

    # H001 correlation
    folds_spy = make_folds(spy, train_years=5, test_years=2)
    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_spy)
    corr = wf.combined_returns.corr(wf_h001.combined_returns.reindex(wf.combined_returns.index))
    print(f"\nH001 correlation: {corr:+.4f}")

    decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="150",
        name="momentum-long-only",
        strategy_class="MomentumLongOnly",
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
        notes="Cross-sectional 12-1 momentum, top quintile, equal-weight, monthly rebalance, 2.5bps/turn.",
        canonical_params={
            "long_days": 252,
            "skip_days": 21,
            "top_quantile": 0.20,
            "monthly_rebalance": True,
        },
    )


if __name__ == "__main__":
    main()
