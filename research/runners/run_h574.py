"""H574: QuarterlyNIMomentum -- PEAD proxy via quarterly NI change.

Cross-sectional: long top quintile by scaled NI change (delta_NI/TA).
Quarterly rebalance (63d). EDGAR data available from ~2009; folds 0-5 zero.

Run: uv run python scripts/run_h574.py
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
from erised.strategies.cross_sectional import QuarterlyNIMomentum
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEP = "=" * 64


def main() -> None:
    print(f"H574: QuarterlyNIMomentum (PEAD proxy)\n{SEP}")

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
    pt = panel.columns.tolist()
    net_income = load_fundamentals_panel(pt, "net_income", panel.index)
    total_assets = load_fundamentals_panel(pt, "total_assets", panel.index)
    print(f"Data: {panel.shape[0]} days, {panel.shape[1]} tickers, {len(folds)} folds")

    wf = walk_forward_cross_sectional(
        panel,
        membership,
        lambda tp, tm: QuarterlyNIMomentum(
            net_income=net_income,
            total_assets=total_assets,
            sectors=sectors,
        ),
        folds=folds,
        spy_returns=spy_rets,
    )
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print(f"\nCanonical: Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}")

    print("\nSensitivity -- rebalance_period (days):")
    for rp in [21, 63, 126]:
        wf_s = walk_forward_cross_sectional(
            panel,
            membership,
            lambda tp, tm, r=rp: QuarterlyNIMomentum(
                net_income=net_income,
                total_assets=total_assets,
                sectors=sectors,
                rebalance_period=r,
            ),
            folds=folds,
            spy_returns=spy_rets,
        )
        lo_s, hi_s = bootstrap_sharpe_ci(wf_s.combined_returns, n_boot=1000)
        print(f"  {rp:>3}d  Sharpe {wf_s.combined_report.sharpe:.3f}  CI [{lo_s:.3f}, {hi_s:.3f}]  MaxDD {wf_s.combined_report.max_drawdown:.2%}")

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
        hypothesis_id="574",
        name="quarterly-ni-momentum",
        strategy_class="QuarterlyNIMomentum",
        family="Set8 Batch2 CS",
        sharpe=rep.sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=rep.cagr,
        max_drawdown=rep.max_drawdown,
        n_trades=rep.n_trades,
        h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results],
        decision=decision,
        notes="PEAD proxy: top quintile by delta(NI)/TA (positive NI surprises). Quarterly rebalance. EDGAR ~2009; folds 0-5 zero.",
        canonical_params={"rebalance_period": 63, "within_sector_quantile": 0.20},
    )


if __name__ == "__main__":
    main()
