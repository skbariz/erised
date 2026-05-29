"""H365: ShrunkSignalComposite -- clipped z-score blend of batch-3 signals.

Run: uv run python scripts/run_h365.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from write_summary import write_summary

from quant.data.fundamentals import load_fundamentals_panel
from quant.data.loader import load_research
from quant.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import MovingAverageCrossover
from quant.strategies.cross_sectional import MomentumLongOnly, ShrunkSignalComposite
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEP = "=" * 64


def main() -> None:
    print(f"H365: ShrunkSignalComposite (clipped z-score batch-3 ensemble)\n{SEP}")
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
        lambda tp, tm: ShrunkSignalComposite(
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

    print(f"\nCanonical (clip_val=2.0, 4 signals, top 20%, quarterly):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}")
    print(f"\nH150 baseline: Sharpe {wf_base.combined_report.sharpe:.3f}")

    print("\nSensitivity -- clip_val:")
    for cv in [1.0, 2.0, 3.0]:
        r = walk_forward_cross_sectional(
            panel,
            membership,
            lambda tp, tm, _cv=cv: ShrunkSignalComposite(
                gross_profit=gross_profit, total_assets=total_assets, net_income=net_income,
                book_equity=book_equity, operating_cf=operating_cf, sectors=sectors, clip_val=_cv
            ),
            folds=folds,
            spy_returns=spy_rets,
        )
        lo_s, hi_s = bootstrap_sharpe_ci(r.combined_returns, n_boot=1000)
        print(f"  clip_val={cv}  Sharpe {r.combined_report.sharpe:.3f}  CI [{lo_s:.3f}, {hi_s:.3f}]  MaxDD {r.combined_report.max_drawdown:.2%}")

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
        hypothesis_id="365",
        name="shrunk-signal-composite",
        strategy_class="ShrunkSignalComposite",
        family="nonlinear-characteristic",
        sharpe=rep.sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=rep.cagr,
        max_drawdown=rep.max_drawdown,
        n_trades=rep.n_trades,
        h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results],
        decision=decision,
        notes="Batch-3 ensemble: clip(z(GP_bin))+clip(z(votes))+clip(z(top_mom))+clip(z(GP/TA)). Kozak et al. 2020.",
        canonical_params={"clip_val": 2.0, "n_bins": 5, "vote_quantile": 0.20, "rebalance_period": 63},
    )


if __name__ == "__main__":
    main()
