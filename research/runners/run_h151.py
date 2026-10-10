"""H151: MomentumMarketNeutral — long top quintile, short bottom quintile.

Run: uv run python scripts/run_h151.py

Note: A 50 bps/year borrow cost drag is added post-backtest to model short
borrow on the short leg. The panel engine already applies 2.5 bps/turn costs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from erised.data.loader import load_research
from erised.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies import MovingAverageCrossover
from erised.strategies.cross_sectional import MomentumMarketNeutral
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional

SEP = "=" * 64

BORROW_COST_ANNUAL = 0.0050  # 50 bps/year on gross short leg (0.5 gross exposure)
TRADING_DAYS = 252


def apply_borrow_cost(returns, gross_short_exposure: float = 0.5) -> object:
    """Deduct daily borrow cost from net returns."""
    daily_borrow = gross_short_exposure * BORROW_COST_ANNUAL / TRADING_DAYS
    return returns - daily_borrow


def main() -> None:
    print(f"H151: MomentumMarketNeutral (12-1m, L/S, top/bottom quintile)\n{SEP}")

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
        lambda tp, tm: MomentumMarketNeutral(monthly_rebalance=True),
        folds=folds,
        spy_returns=spy_rets,
    )
    net_after_borrow = apply_borrow_cost(wf.combined_returns)

    import erised.metrics.performance as perf_mod

    rep = perf_mod.report(net_after_borrow)
    lo, hi = bootstrap_sharpe_ci(net_after_borrow, n_boot=5000)

    print("\nH151 Canonical (after 50bps/yr borrow cost):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}")

    print("\nSensitivity -- top/bottom quantile:")
    for q in [0.10, 0.20, 0.30]:
        w = walk_forward_cross_sectional(
            panel,
            membership,
            lambda tp, tm, _q=q: MomentumMarketNeutral(top_quantile=_q, bottom_quantile=_q),
            folds=folds,
            spy_returns=spy_rets,
        )
        adj = apply_borrow_cost(w.combined_returns)
        lo_q, hi_q = bootstrap_sharpe_ci(adj, n_boot=1000)
        r = perf_mod.report(adj)
        print(
            f"  quantile={q:.0%}  Sharpe {r.sharpe:.3f}"
            f"  CI [{lo_q:.3f}, {hi_q:.3f}]  MaxDD {r.max_drawdown:.2%}"
        )

    print("\nPer-fold Sharpe (after borrow cost):")
    for i, fold_result in enumerate(wf.fold_results):
        fold_adj = apply_borrow_cost(fold_result.net_returns)
        print(
            f"  Fold {i:>2}: {sharpe(fold_adj):+.3f}"
            f"  ({folds[i].test_start.date()} -- {folds[i].test_end.date()})"
        )

    folds_spy = make_folds(spy, train_years=5, test_years=2)
    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_spy)
    corr = net_after_borrow.corr(wf_h001.combined_returns.reindex(net_after_borrow.index))
    print(f"\nH001 correlation: {corr:+.4f}")

    decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="151",
        name="momentum-market-neutral",
        strategy_class="MomentumMarketNeutral",
        family="cross-sectional momentum",
        sharpe=rep.sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=rep.cagr,
        max_drawdown=rep.max_drawdown,
        n_trades=rep.n_trades,
        h001_corr=float(corr),
        fold_sharpes=[float(sharpe(apply_borrow_cost(f.net_returns))) for f in wf.fold_results],
        decision=decision,
        notes="L/S 0.5 gross each leg. After 50bps/yr borrow + 2.5bps/turn. ~zero market beta.",
        canonical_params={
            "top_quantile": 0.20,
            "bottom_quantile": 0.20,
            "monthly_rebalance": True,
            "borrow_bps_annual": 50,
        },
    )


if __name__ == "__main__":
    main()
