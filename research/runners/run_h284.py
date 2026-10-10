"""H284: LowCoverageMomentumProxy -- momentum minus idio-vol spike ratio.

Run: uv run python scripts/run_h284.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from write_summary import write_summary

from erised.data.loader import load_research
from erised.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies.base import MovingAverageCrossover
from erised.strategies.cross_sectional import LowCoverageMomentumProxy, MomentumLongOnly
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEP = "=" * 64


def main() -> None:
    print(f"H284: LowCoverageMomentumProxy (momentum - idio-vol-ratio, top 20%)\n{SEP}")

    membership = load_sp500_members()
    tickers = get_all_historical_tickers()
    print(f"Loading prices for {len(tickers)} historical tickers...")
    panel = load_stocks_panel(tickers)

    common_idx = panel.index.intersection(membership.index)
    panel = panel.loc[common_idx]
    membership = membership.loc[common_idx]

    spy = load_research("SPY")
    spy_prices = spy["adj_close"]
    spy_rets = spy_prices.pct_change().fillna(0.0)

    sectors = pd.read_parquet(DATA_DIR / "sp500_sectors.parquet")["sector"]
    folds = make_folds(panel, train_years=5, test_years=2)
    print(f"Data: {panel.shape[0]} days, {panel.shape[1]} tickers, {sectors.nunique()} sectors")
    print(f"Folds: {len(folds)}")

    wf = walk_forward_cross_sectional(
        panel,
        membership,
        lambda tp, tm: LowCoverageMomentumProxy(spy_prices=spy_prices, sectors=sectors),
        folds=folds,
        spy_returns=spy_rets,
    )
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)

    wf_h150 = walk_forward_cross_sectional(
        panel, membership, lambda tp, tm: MomentumLongOnly(), folds=folds, spy_returns=spy_rets
    )
    rep150 = wf_h150.combined_report

    print("\nCanonical (mom_window=252, skip=21, beta_window=63, top 20%):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}")
    print(f"\nH150 baseline: Sharpe {rep150.sharpe:.3f}  MaxDD {rep150.max_drawdown:.2%}")

    print("\nSensitivity -- beta_window:")
    for bw in [42, 63, 126]:
        r = walk_forward_cross_sectional(
            panel,
            membership,
            lambda tp, tm, _bw=bw: LowCoverageMomentumProxy(
                spy_prices=spy_prices, sectors=sectors, beta_window=_bw
            ),
            folds=folds,
            spy_returns=spy_rets,
        )
        lo_s, hi_s = bootstrap_sharpe_ci(r.combined_returns, n_boot=1000)
        print(
            f"  beta_window={bw}  Sharpe {r.combined_report.sharpe:.3f}"
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
        hypothesis_id="284",
        name="low-coverage-momentum",
        strategy_class="LowCoverageMomentumProxy",
        family="cross-sectional momentum lifecycle",
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
            "Hong-Lim-Stein (2000) low-coverage momentum proxy. "
            "z(mom_12_1) - z(idio_vol_ratio) where idio_vol_ratio = ivol_63d/ivol_252d. "
            "Idio-vol from rolling OLS beta on SPY. Low ratio = quiet information env. "
            "Distinct from H271 (total vol level). Sector-neutral top 20%, monthly rebalance."
        ),
        canonical_params={
            "mom_window": 252,
            "skip_window": 21,
            "beta_window": 63,
            "ivol_short": 63,
            "ivol_long": 252,
            "within_sector_quantile": 0.20,
        },
    )


if __name__ == "__main__":
    main()
