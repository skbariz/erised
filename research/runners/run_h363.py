"""H363: ElasticNetReturnForecast -- regularized ML cross-sectional ranker.

Run: uv run python scripts/run_h363.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from write_summary import write_summary

from erised.data.fundamentals import load_fundamentals_panel
from erised.data.loader import load_research
from erised.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies import MovingAverageCrossover
from erised.strategies.cross_sectional import ElasticNetReturnForecast, MomentumLongOnly
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEP = "=" * 64


def main() -> None:
    print(f"H363: ElasticNetReturnForecast (regularized ML cross-sectional ranker)\n{SEP}")
    membership = load_sp500_members()
    tickers = get_all_historical_tickers()
    panel = load_stocks_panel(tickers)
    common_idx = panel.index.intersection(membership.index)
    panel = panel.loc[common_idx]
    membership = membership.loc[common_idx]
    spy = load_research("SPY")
    spy_rets = spy["adj_close"].pct_change().fillna(0.0)
    sectors = pd.read_parquet(DATA_DIR / "sp500_sectors.parquet")["sector"]
    folds = make_folds(panel, train_years=5, test_years=2)
    print("Loading fundamental data...")
    panel_tickers = panel.columns.tolist()
    gross_profit = load_fundamentals_panel(panel_tickers, "gross_profit", panel.index)
    total_assets = load_fundamentals_panel(panel_tickers, "total_assets", panel.index)
    net_income = load_fundamentals_panel(panel_tickers, "net_income", panel.index)
    book_equity = load_fundamentals_panel(panel_tickers, "book_equity", panel.index)
    operating_cf = load_fundamentals_panel(panel_tickers, "operating_cf", panel.index)
    print(f"Data: {panel.shape[0]} days, {panel.shape[1]} tickers  Folds: {len(folds)}")

    wf = walk_forward_cross_sectional(
        panel,
        membership,
        lambda tp, tm: ElasticNetReturnForecast(
            gross_profit=gross_profit,
            total_assets=total_assets,
            net_income=net_income,
            book_equity=book_equity,
            operating_cf=operating_cf,
            sectors=sectors,
        ),
        folds=folds,
        spy_returns=spy_rets,
    )
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)

    wf_base = walk_forward_cross_sectional(
        panel, membership, lambda tp, tm: MomentumLongOnly(), folds=folds, spy_returns=spy_rets
    )

    print(f"\nCanonical (alpha=0.01, l1_ratio=0.5, train_window=756d, quarterly):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}")
    print(f"\nH150 baseline: Sharpe {wf_base.combined_report.sharpe:.3f}")

    print("\nSensitivity -- alpha (regularization strength):")
    for a in [0.001, 0.01, 0.1]:
        r = walk_forward_cross_sectional(
            panel,
            membership,
            lambda tp, tm, _a=a: ElasticNetReturnForecast(
                gross_profit=gross_profit, total_assets=total_assets, net_income=net_income,
                book_equity=book_equity, operating_cf=operating_cf, sectors=sectors, alpha=_a
            ),
            folds=folds,
            spy_returns=spy_rets,
        )
        lo_s, hi_s = bootstrap_sharpe_ci(r.combined_returns, n_boot=1000)
        print(f"  alpha={a}  Sharpe {r.combined_report.sharpe:.3f}  CI [{lo_s:.3f}, {hi_s:.3f}]  MaxDD {r.combined_report.max_drawdown:.2%}")

    print("\nPer-fold Sharpe:")
    for i, fold_result in enumerate(wf.fold_results):
        print(f"  Fold {i:>2}: {sharpe(fold_result.net_returns):+.3f}  ({folds[i].test_start.date()} -- {folds[i].test_end.date()})")

    folds_spy = make_folds(spy, train_years=5, test_years=2)
    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_spy)
    corr = wf.combined_returns.corr(wf_h001.combined_returns.reindex(wf.combined_returns.index))
    print(f"\nH001 correlation: {corr:+.4f}")

    decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="363",
        name="elastic-net-return-forecast",
        strategy_class="ElasticNetReturnForecast",
        family="ml-characteristic",
        sharpe=rep.sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=rep.cagr,
        max_drawdown=rep.max_drawdown,
        n_trades=rep.n_trades,
        h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results],
        decision=decision,
        notes="ElasticNet cross-sectional ranker on 6 features (Freyberger+Kozak 2020). Quarterly rebalance.",
        canonical_params={"alpha": 0.01, "l1_ratio": 0.5, "train_window": 756, "rebalance_period": 63},
    )


if __name__ == "__main__":
    main()
