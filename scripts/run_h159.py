"""H159: SmoothVolTargetMomentum — smooth signal (H155) + vol-target (H157) composite.

Run: uv run python scripts/run_h159.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from quant.data.loader import load_research
from quant.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import MovingAverageCrossover
from quant.strategies.cross_sectional import (
    MomentumLongOnly,
    SmoothMomentum,
    SmoothVolTargetMomentum,
    VolScaledMomentum,
)
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional

SEP = "=" * 64


def main() -> None:
    print(f"H159: SmoothVolTargetMomentum (H155 signal + H157 vol-target)\n{SEP}")

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
        lambda tp, tm: SmoothVolTargetMomentum(vol_target=0.12),
        folds=folds,
        spy_returns=spy_rets,
    )
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)

    print("\nH159 Canonical (smooth signal + vol_target=12%):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}")

    # Compare components
    wf_h150 = walk_forward_cross_sectional(
        panel, membership, lambda tp, tm: MomentumLongOnly(), folds=folds, spy_returns=spy_rets
    )
    wf_h155 = walk_forward_cross_sectional(
        panel, membership, lambda tp, tm: SmoothMomentum(), folds=folds, spy_returns=spy_rets
    )
    wf_h157 = walk_forward_cross_sectional(
        panel,
        membership,
        lambda tp, tm: VolScaledMomentum(vol_target=0.12),
        folds=folds,
        spy_returns=spy_rets,
    )
    print(
        f"\nH150 (raw): Sharpe {wf_h150.combined_report.sharpe:.3f}"
        f"  MaxDD {wf_h150.combined_report.max_drawdown:.2%}"
    )
    print(
        f"H155 (smooth only): Sharpe {wf_h155.combined_report.sharpe:.3f}"
        f"  MaxDD {wf_h155.combined_report.max_drawdown:.2%}"
    )
    print(
        f"H157 (vol-scale only): Sharpe {wf_h157.combined_report.sharpe:.3f}"
        f"  MaxDD {wf_h157.combined_report.max_drawdown:.2%}"
    )

    print("\nSensitivity -- vol_target:")
    for vt in [0.08, 0.12, 0.16]:
        w = walk_forward_cross_sectional(
            panel,
            membership,
            lambda tp, tm, _vt=vt: SmoothVolTargetMomentum(vol_target=_vt),
            folds=folds,
            spy_returns=spy_rets,
        )
        lo_v, hi_v = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  vol_target={vt:.0%}  Sharpe {w.combined_report.sharpe:.3f}"
            f"  CI [{lo_v:.3f}, {hi_v:.3f}]  MaxDD {w.combined_report.max_drawdown:.2%}"
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
        hypothesis_id="159",
        name="smooth-vol-target-momentum",
        strategy_class="SmoothVolTargetMomentum",
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
        notes="H155 smooth signal + H157 vol-target. Dual protection: signal quality + exposure mgmt.",
        canonical_params={
            "long_days": 252,
            "skip_days": 21,
            "top_quantile": 0.20,
            "vol_target": 0.12,
            "vol_window": 21,
            "monthly_rebalance": True,
        },
    )


if __name__ == "__main__":
    main()
