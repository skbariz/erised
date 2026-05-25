"""H162: CompositeValueMomentum — 12-1m momentum + 3yr reversal z-score blend.

Run: uv run python scripts/run_h162.py
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
    CompositeValueMomentum,
    LongTermReversal,
    MomentumLongOnly,
)
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional

SEP = "=" * 64


def main() -> None:
    print(f"H162: CompositeValueMomentum (12-1m momentum + 3yr reversal z-score blend)\n{SEP}")

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
        lambda tp, tm: CompositeValueMomentum(monthly_rebalance=True),
        folds=folds,
        spy_returns=spy_rets,
    )
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)

    print("\nH162 Canonical (50/50 z-score blend, reversal_days=756):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}")

    # Compare pure components
    wf_h150 = walk_forward_cross_sectional(
        panel, membership, lambda tp, tm: MomentumLongOnly(), folds=folds, spy_returns=spy_rets
    )
    wf_h160 = walk_forward_cross_sectional(
        panel, membership, lambda tp, tm: LongTermReversal(), folds=folds, spy_returns=spy_rets
    )
    print(
        f"\nH150 (momentum only): Sharpe {wf_h150.combined_report.sharpe:.3f}"
        f"  MaxDD {wf_h150.combined_report.max_drawdown:.2%}"
    )
    print(
        f"H160 (reversal only): Sharpe {wf_h160.combined_report.sharpe:.3f}"
        f"  MaxDD {wf_h160.combined_report.max_drawdown:.2%}"
    )

    print("\nSensitivity -- reversal_days:")
    for rd in [504, 756, 1008]:
        w = walk_forward_cross_sectional(
            panel,
            membership,
            lambda tp, tm, _rd=rd: CompositeValueMomentum(reversal_days=_rd),
            folds=folds,
            spy_returns=spy_rets,
        )
        lo_r, hi_r = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        months = rd // 21
        print(
            f"  reversal_days={rd} (~{months}m)  Sharpe {w.combined_report.sharpe:.3f}"
            f"  CI [{lo_r:.3f}, {hi_r:.3f}]  MaxDD {w.combined_report.max_drawdown:.2%}"
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
        hypothesis_id="162",
        name="composite-value-momentum",
        strategy_class="CompositeValueMomentum",
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
        notes="50/50 z-score blend: 12-1m momentum + negated 36m return (Asness-Moskowitz-Pedersen 2013 style).",
        canonical_params={
            "long_days": 252,
            "skip_days": 21,
            "reversal_days": 756,
            "top_quantile": 0.20,
            "monthly_rebalance": True,
        },
    )


if __name__ == "__main__":
    main()
