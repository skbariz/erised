"""H291: LowTurnoverValueFactor -- earnings yield with buy/hold band.

Run: uv run python scripts/run_h291.py
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
from erised.strategies.cross_sectional import LowTurnoverValueFactor, MomentumLongOnly
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEP = "=" * 64


def main() -> None:
    print(f"H291: LowTurnoverValueFactor (earnings yield + buy/hold band)\n{SEP}")

    membership = load_sp500_members()
    tickers = get_all_historical_tickers()
    print(f"Loading prices for {len(tickers)} historical tickers...")
    panel = load_stocks_panel(tickers)

    common_idx = panel.index.intersection(membership.index)
    panel = panel.loc[common_idx]
    membership = membership.loc[common_idx]

    spy = load_research("SPY")
    spy_rets = spy["adj_close"].pct_change().fillna(0.0)

    sectors = pd.read_parquet(DATA_DIR / "sp500_sectors.parquet")["sector"]
    panel_tickers = panel.columns.tolist()
    ni = load_fundamentals_panel(panel_tickers, "net_income", panel.index)
    sh = load_fundamentals_panel(panel_tickers, "shares_outstanding", panel.index)

    folds = make_folds(panel, train_years=5, test_years=2)
    print(f"Data: {panel.shape[0]} days, {panel.shape[1]} tickers, {sectors.nunique()} sectors")
    print(f"Folds: {len(folds)}")

    wf = walk_forward_cross_sectional(
        panel,
        membership,
        lambda tp, tm: LowTurnoverValueFactor(
            net_income=ni, shares_outstanding=sh, sectors=sectors
        ),
        folds=folds,
        spy_returns=spy_rets,
    )
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)

    wf_h150 = walk_forward_cross_sectional(
        panel, membership, lambda tp, tm: MomentumLongOnly(), folds=folds, spy_returns=spy_rets
    )
    rep150 = wf_h150.combined_report

    print("\nCanonical (buy=0.20, hold=0.40, rebal_period=21):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}")
    print(f"\nH150 baseline: Sharpe {rep150.sharpe:.3f}  MaxDD {rep150.max_drawdown:.2%}")

    print("\nSensitivity -- hold_quantile:")
    for hq in [0.30, 0.40, 0.50]:
        r = walk_forward_cross_sectional(
            panel,
            membership,
            lambda tp, tm, _hq=hq: LowTurnoverValueFactor(
                net_income=ni, shares_outstanding=sh, sectors=sectors, hold_quantile=_hq
            ),
            folds=folds,
            spy_returns=spy_rets,
        )
        lo_s, hi_s = bootstrap_sharpe_ci(r.combined_returns, n_boot=1000)
        print(
            f"  hold_quantile={hq:.2f}  Sharpe {r.combined_report.sharpe:.3f}"
            f"  CI [{lo_s:.3f}, {hi_s:.3f}]  MaxDD {r.combined_report.max_drawdown:.2%}"
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
        hypothesis_id="291",
        name="low-turnover-value",
        strategy_class="LowTurnoverValueFactor",
        family="execution-aware",
        sharpe=rep.sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=rep.cagr,
        max_drawdown=rep.max_drawdown,
        n_trades=rep.n_trades,
        h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results],
        decision=decision,
        notes=(
            "Novy-Marx & Velikov (2016) buy/hold band on earnings yield. "
            "Entry: rank >= 0.80 (top 20%). Hold until rank < 0.60 (exit below top 40%). "
            "Equal-weight held stocks; monthly rebalance check. "
            "Reduces round-trip turnover by ~62% vs standard monthly rebalance."
        ),
        canonical_params={
            "buy_quantile": 0.20,
            "hold_quantile": 0.40,
            "rebalance_period": 21,
        },
    )


if __name__ == "__main__":
    main()
