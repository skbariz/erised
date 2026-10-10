"""H575: ThreeFactorDailyReversal -- daily reversal using 3-factor residuals.

Market (SPY) + size (IWM-SPY) + value (IVE-IVW) 3-factor residual reversal.
Long bottom decile (most negative = oversold), short top decile. Dollar-neutral.
Economic grounding vs PCA-based H191/H192 which take 12.9% combined in H559.

Run: uv run python scripts/run_h575.py
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
from erised.strategies.cross_sectional import ThreeFactorDailyReversal
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEP = "=" * 64


def main() -> None:
    print(f"H575: ThreeFactorDailyReversal\n{SEP}")

    membership = load_sp500_members()
    tickers = get_all_historical_tickers()
    panel = load_stocks_panel(tickers)

    common_idx = panel.index.intersection(membership.index)
    panel = panel.loc[common_idx]
    membership = membership.loc[common_idx]

    spy = load_research("SPY")
    iwm = load_research("IWM")[["adj_close"]].rename(columns={"adj_close": "iwm"})
    ive = load_research("IVE")[["adj_close"]].rename(columns={"adj_close": "ive"})
    ivw = load_research("IVW")[["adj_close"]].rename(columns={"adj_close": "ivw"})
    factors = spy[["adj_close"]].rename(columns={"adj_close": "spy"}).join(iwm).join(ive).join(ivw)
    for col in ["iwm", "ive", "ivw"]:
        factors[col] = factors[col].ffill()

    spy_rets = factors["spy"].pct_change().fillna(0.0)
    iwm_rets = factors["iwm"].pct_change().fillna(0.0)
    ive_rets = factors["ive"].pct_change().fillna(0.0)
    ivw_rets = factors["ivw"].pct_change().fillna(0.0)

    # Size factor = IWM - SPY (small-cap premium)
    size_rets = iwm_rets - spy_rets
    # Value factor = IVE - IVW (value minus growth)
    value_rets = ive_rets - ivw_rets

    sectors = pd.read_parquet(DATA_DIR / "sp500_sectors.parquet")["sector"]
    folds = make_folds(panel, train_years=5, test_years=2)
    print(f"Data: {panel.shape[0]} days, {panel.shape[1]} tickers, {len(folds)} folds")

    wf = walk_forward_cross_sectional(
        panel,
        membership,
        lambda tp, tm: ThreeFactorDailyReversal(
            spy_rets=spy_rets,
            size_rets=size_rets,
            value_rets=value_rets,
            sectors=sectors,
        ),
        folds=folds,
        spy_returns=spy_rets,
    )
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print(f"\nCanonical (quantile=10%): Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}")

    print("\nSensitivity -- quantile:")
    for q in [0.05, 0.10, 0.15, 0.20]:
        wf_s = walk_forward_cross_sectional(
            panel,
            membership,
            lambda tp, tm, _q=q: ThreeFactorDailyReversal(
                spy_rets=spy_rets, size_rets=size_rets, value_rets=value_rets,
                sectors=sectors, quantile=_q,
            ),
            folds=folds,
            spy_returns=spy_rets,
        )
        lo_s, hi_s = bootstrap_sharpe_ci(wf_s.combined_returns, n_boot=1000)
        print(f"  q={q:.0%}  Sharpe {wf_s.combined_report.sharpe:.3f}  CI [{lo_s:.3f}, {hi_s:.3f}]  MaxDD {wf_s.combined_report.max_drawdown:.2%}")

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
        hypothesis_id="575",
        name="three-factor-daily-reversal",
        strategy_class="ThreeFactorDailyReversal",
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
        notes="3-factor (mkt+size+val) daily residual reversal. L/S top/bottom 10%. Dollar-neutral. Economic grounding vs PCA-based H191/H192.",
        canonical_params={"quantile": 0.10, "beta_window": 60},
    )


if __name__ == "__main__":
    main()
