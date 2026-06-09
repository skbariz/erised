"""H576: QualityMomentumWeeklyRotation -- quality gate + weekly momentum.

Monthly quality filter (top 40% by GP/A), then weekly selection of top-20
by 21-day return. Long-only, equal-weight, 5-day rebalance.

Run: uv run python scripts/run_h576.py
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
from quant.strategies.cross_sectional import QualityMomentumWeeklyRotation
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEP = "=" * 64


def main() -> None:
    print(f"H576: QualityMomentumWeeklyRotation\n{SEP}")

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
    gross_profit = load_fundamentals_panel(pt, "gross_profit", panel.index)
    total_assets = load_fundamentals_panel(pt, "total_assets", panel.index)
    print(f"Data: {panel.shape[0]} days, {panel.shape[1]} tickers, {len(folds)} folds")

    wf = walk_forward_cross_sectional(
        panel,
        membership,
        lambda tp, tm: QualityMomentumWeeklyRotation(
            gross_profit=gross_profit,
            total_assets=total_assets,
            sectors=sectors,
        ),
        folds=folds,
        spy_returns=spy_rets,
    )
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print(f"\nCanonical (top_n=20, quality=40%, mom=21d): Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}")

    print("\nSensitivity -- top_n:")
    for tn in [10, 20, 30]:
        wf_s = walk_forward_cross_sectional(
            panel, membership,
            lambda tp, tm, t=tn: QualityMomentumWeeklyRotation(
                gross_profit=gross_profit, total_assets=total_assets,
                sectors=sectors, top_n=t,
            ),
            folds=folds, spy_returns=spy_rets,
        )
        lo_s, hi_s = bootstrap_sharpe_ci(wf_s.combined_returns, n_boot=1000)
        print(f"  top_n={tn}  Sharpe {wf_s.combined_report.sharpe:.3f}  CI [{lo_s:.3f}, {hi_s:.3f}]  MaxDD {wf_s.combined_report.max_drawdown:.2%}")

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
        hypothesis_id="576",
        name="quality-momentum-weekly-rotation",
        strategy_class="QualityMomentumWeeklyRotation",
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
        notes="Quality gate (top 40% GP/A) + weekly momentum (21d, top-20). 5-day rebalance. Long-only.",
        canonical_params={"quality_quantile": 0.40, "top_n": 20, "mom_window": 21, "rebalance_period": 5},
    )


if __name__ == "__main__":
    main()
