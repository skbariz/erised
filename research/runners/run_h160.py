"""H160: LargeCapValue — Book-to-Market + Earnings Yield composite.

Run: uv run python scripts/run_h160.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from erised.data.fundamentals import load_fundamentals_panel
from erised.data.loader import load_research
from erised.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies.base import MovingAverageCrossover
from erised.strategies.cross_sectional import LargeCapValue, MomentumLongOnly
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional

SEP = "=" * 64


def main() -> None:
    print(f"H160: LargeCapValue (B/M + E/P composite, FF1993 style)\n{SEP}")

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

    print("Loading fundamental data (book_equity, net_income, shares_outstanding)...")
    panel_tickers = panel.columns.tolist()
    book_equity = load_fundamentals_panel(panel_tickers, "book_equity", panel.index)
    net_income = load_fundamentals_panel(panel_tickers, "net_income", panel.index)
    shares = load_fundamentals_panel(panel_tickers, "shares_outstanding", panel.index)

    fund_coverage = book_equity.notna().any(axis=1).sum()
    print(f"Fundamental data coverage: {fund_coverage}/{len(panel.index)} dates with >=1 stock")

    wf = walk_forward_cross_sectional(
        panel,
        membership,
        lambda tp, tm: LargeCapValue(
            book_equity=book_equity,
            net_income=net_income,
            shares_outstanding=shares,
        ),
        folds=folds,
        spy_returns=spy_rets,
    )
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)

    print("\nH160 Canonical (B/M + E/P, top_quantile=20%):")
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

    print("\nSensitivity -- top_quantile:")
    for tq in [0.10, 0.20, 0.30]:
        w = walk_forward_cross_sectional(
            panel,
            membership,
            lambda tp, tm, _tq=tq: LargeCapValue(
                book_equity=book_equity,
                net_income=net_income,
                shares_outstanding=shares,
                top_quantile=_tq,
            ),
            folds=folds,
            spy_returns=spy_rets,
        )
        lo_t, hi_t = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  top_quantile={tq:.0%}  Sharpe {w.combined_report.sharpe:.3f}"
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
        hypothesis_id="160",
        name="large-cap-value",
        strategy_class="LargeCapValue",
        family="cross-sectional value",
        sharpe=rep.sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=rep.cagr,
        max_drawdown=rep.max_drawdown,
        n_trades=rep.n_trades,
        h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results],
        decision=decision,
        notes="B/M + E/P composite value. Point-in-time EDGAR XBRL fundamentals. Coverage ~2009+.",
        canonical_params={"top_quantile": 0.20, "monthly_rebalance": True},
    )


if __name__ == "__main__":
    main()
