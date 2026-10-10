"""H163: SectorNeutralValueMomentum — value+momentum ranked within GICS sectors.

Run: uv run python scripts/run_h163.py
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
from erised.strategies.base import MovingAverageCrossover
from erised.strategies.cross_sectional import MomentumLongOnly, SectorNeutralValueMomentum
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional

SEP = "=" * 64

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def main() -> None:
    print(f"H163: SectorNeutralValueMomentum (B/M+E/P+momentum, sector-neutral)\n{SEP}")

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

    sectors_path = DATA_DIR / "sp500_sectors.parquet"
    sectors = pd.read_parquet(sectors_path)["sector"]
    print(f"Sector data: {len(sectors)} tickers, {sectors.nunique()} sectors")

    print("Loading fundamental data (book_equity, net_income, shares_outstanding)...")
    panel_tickers = panel.columns.tolist()
    book_equity = load_fundamentals_panel(panel_tickers, "book_equity", panel.index)
    net_income = load_fundamentals_panel(panel_tickers, "net_income", panel.index)
    shares = load_fundamentals_panel(panel_tickers, "shares_outstanding", panel.index)

    wf = walk_forward_cross_sectional(
        panel,
        membership,
        lambda tp, tm: SectorNeutralValueMomentum(
            book_equity=book_equity,
            net_income=net_income,
            shares_outstanding=shares,
            sectors=sectors,
        ),
        folds=folds,
        spy_returns=spy_rets,
    )
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)

    print("\nH163 Canonical (sector-neutral V+M, within_sector_quantile=30%):")
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

    print("\nSensitivity -- within_sector_quantile:")
    for wq in [0.20, 0.30, 0.40]:
        w = walk_forward_cross_sectional(
            panel,
            membership,
            lambda tp, tm, _wq=wq: SectorNeutralValueMomentum(
                book_equity=book_equity,
                net_income=net_income,
                shares_outstanding=shares,
                sectors=sectors,
                within_sector_quantile=_wq,
            ),
            folds=folds,
            spy_returns=spy_rets,
        )
        lo_w, hi_w = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  within_sector_quantile={wq:.0%}  Sharpe {w.combined_report.sharpe:.3f}"
            f"  CI [{lo_w:.3f}, {hi_w:.3f}]  MaxDD {w.combined_report.max_drawdown:.2%}"
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
        hypothesis_id="163",
        name="sector-neutral-value-momentum",
        strategy_class="SectorNeutralValueMomentum",
        family="cross-sectional value+momentum (sector-neutral)",
        sharpe=rep.sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=rep.cagr,
        max_drawdown=rep.max_drawdown,
        n_trades=rep.n_trades,
        h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results],
        decision=decision,
        notes="Sector-neutral B/M+E/P+12-1m momentum composite. Top 30% per GICS sector. Israel-Moskowitz 2013.",
        canonical_params={
            "long_days": 252,
            "skip_days": 21,
            "within_sector_quantile": 0.30,
            "monthly_rebalance": True,
        },
    )


if __name__ == "__main__":
    main()
