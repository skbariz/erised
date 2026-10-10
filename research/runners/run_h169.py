"""H169: QualityValueComposite — (1/3)*z(B/M) + (1/3)*z(GP/A) + (1/3)*z(-accruals), sector-neutral.

Combines value (B/M), gross profitability, and low accruals in one composite.
Run: uv run python scripts/run_h169.py
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
from erised.strategies.cross_sectional import MomentumLongOnly, QualityValueComposite
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEP = "=" * 64


def main() -> None:
    print(f"H169: QualityValueComposite (B/M + GP/A + low accruals, sector-neutral)\n{SEP}")

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

    print(
        "Loading fundamental data (book_equity, shares_outstanding, gross_profit, total_assets, net_income, operating_cf)..."
    )
    panel_tickers = panel.columns.tolist()
    book_equity = load_fundamentals_panel(panel_tickers, "book_equity", panel.index)
    shares = load_fundamentals_panel(panel_tickers, "shares_outstanding", panel.index)
    gross_profit = load_fundamentals_panel(panel_tickers, "gross_profit", panel.index)
    total_assets = load_fundamentals_panel(panel_tickers, "total_assets", panel.index)
    net_income = load_fundamentals_panel(panel_tickers, "net_income", panel.index)
    operating_cf = load_fundamentals_panel(panel_tickers, "operating_cf", panel.index)

    fund_coverage = book_equity.notna().any(axis=1).sum()
    print(f"Book equity coverage: {fund_coverage}/{len(panel.index)} dates with >=1 stock")

    sectors = pd.read_parquet(DATA_DIR / "sp500_sectors.parquet")["sector"]
    print(f"Sector data: {len(sectors)} tickers, {sectors.nunique()} sectors")

    wf = walk_forward_cross_sectional(
        panel,
        membership,
        lambda tp, tm: QualityValueComposite(
            book_equity=book_equity,
            shares_outstanding=shares,
            gross_profit=gross_profit,
            total_assets=total_assets,
            net_income=net_income,
            operating_cf=operating_cf,
            sectors=sectors,
        ),
        folds=folds,
        spy_returns=spy_rets,
    )
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)

    print("\nH169 Canonical (sector-neutral, within_sector_quantile=20%):")
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
    for q in [0.10, 0.20, 0.30]:
        w = walk_forward_cross_sectional(
            panel,
            membership,
            lambda tp, tm, _q=q: QualityValueComposite(
                book_equity=book_equity,
                shares_outstanding=shares,
                gross_profit=gross_profit,
                total_assets=total_assets,
                net_income=net_income,
                operating_cf=operating_cf,
                sectors=sectors,
                within_sector_quantile=_q,
            ),
            folds=folds,
            spy_returns=spy_rets,
        )
        lo_s, hi_s = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  within_sector_quantile={q:.0%}  Sharpe {w.combined_report.sharpe:.3f}"
            f"  CI [{lo_s:.3f}, {hi_s:.3f}]  MaxDD {w.combined_report.max_drawdown:.2%}"
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
        hypothesis_id="169",
        name="quality-value-composite",
        strategy_class="QualityValueComposite",
        family="cross-sectional quality+value (composite)",
        sharpe=rep.sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=rep.cagr,
        max_drawdown=rep.max_drawdown,
        n_trades=rep.n_trades,
        h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results],
        decision=decision,
        notes="(1/3)*z(B/M) + (1/3)*z(GP/A) + (1/3)*z(-accruals). Combines value, profitability, and earnings quality. Sector-neutral.",
        canonical_params={
            "within_sector_quantile": 0.20,
            "monthly_rebalance": True,
        },
    )


if __name__ == "__main__":
    main()
