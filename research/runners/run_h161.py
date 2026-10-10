"""H161: QualityFilteredReversal — 5-year losers with quality gate.

Run: uv run python scripts/run_h161.py
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
from erised.strategies.cross_sectional import MomentumLongOnly, QualityFilteredReversal
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional

SEP = "=" * 64


def main() -> None:
    print(f"H161: QualityFilteredReversal (5-yr losers + quality gate)\n{SEP}")

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

    print("Loading fundamental data (operating_cf, current_assets, current_liabilities)...")
    panel_tickers = panel.columns.tolist()
    operating_cf = load_fundamentals_panel(panel_tickers, "operating_cf", panel.index)
    current_assets = load_fundamentals_panel(panel_tickers, "current_assets", panel.index)
    current_liabilities = load_fundamentals_panel(panel_tickers, "current_liabilities", panel.index)

    wf = walk_forward_cross_sectional(
        panel,
        membership,
        lambda tp, tm: QualityFilteredReversal(
            operating_cf=operating_cf,
            current_assets=current_assets,
            current_liabilities=current_liabilities,
        ),
        folds=folds,
        spy_returns=spy_rets,
    )
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)

    print("\nH161 Canonical (long_days=1260, recovery_days=126, quality gate):")
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

    print("\nSensitivity -- long_days (reversal horizon):")
    for ld in [756, 1008, 1260]:
        w = walk_forward_cross_sectional(
            panel,
            membership,
            lambda tp, tm, _ld=ld: QualityFilteredReversal(
                operating_cf=operating_cf,
                current_assets=current_assets,
                current_liabilities=current_liabilities,
                long_days=_ld,
            ),
            folds=folds,
            spy_returns=spy_rets,
        )
        lo_l, hi_l = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        months = ld // 21
        print(
            f"  long_days={ld} (~{months}m)  Sharpe {w.combined_report.sharpe:.3f}"
            f"  CI [{lo_l:.3f}, {hi_l:.3f}]  MaxDD {w.combined_report.max_drawdown:.2%}"
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
        hypothesis_id="161",
        name="quality-filtered-reversal",
        strategy_class="QualityFilteredReversal",
        family="cross-sectional contrarian/reversal",
        sharpe=rep.sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=rep.cagr,
        max_drawdown=rep.max_drawdown,
        n_trades=rep.n_trades,
        h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results],
        decision=decision,
        notes="5-yr price losers + quality gate (OCF>0, current ratio>1, 6m recovery). De Bondt-Thaler upgraded.",
        canonical_params={
            "long_days": 1260,
            "skip_days": 21,
            "recovery_days": 126,
            "top_quantile": 0.20,
            "monthly_rebalance": True,
        },
    )


if __name__ == "__main__":
    main()
