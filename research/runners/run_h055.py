"""H055: FiveAssetRotation -- standalone walk-forward runner.

Run: uv run python scripts/run_h055.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies.base import FiveAssetRotation, MovingAverageCrossover
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    gld = load_research("GLD")[["adj_close"]].rename(columns={"adj_close": "gld_close"})
    dbc = load_research("DBC")[["adj_close"]].rename(columns={"adj_close": "dbc_close"})
    vnq = load_research("VNQ")[["adj_close"]].rename(columns={"adj_close": "vnq_close"})
    prices = (
        spy.join(tlt, how="left").join(gld, how="left").join(dbc, how="left").join(vnq, how="left")
    )
    for col in ["tlt_close", "gld_close", "dbc_close", "vnq_close"]:
        prices[col] = prices[col].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H055: FiveAssetRotation (canonical long_days=252, skip_days=21)\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward_multi(prices, lambda _: FiveAssetRotation(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print("\nCanonical (long_days=252, skip_days=21):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    print("\nSensitivity -- long_days (skip_days=21):")
    for ld in [126, 189, 252, 315]:
        w = walk_forward_multi(
            prices, lambda _, ld_=ld: FiveAssetRotation(long_days=ld_, skip_days=21), folds=folds
        )
        lo_v, hi_v = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  long={ld}d  Sharpe {w.combined_report.sharpe:.3f}"
            f"  CI [{lo_v:.3f}, {hi_v:.3f}]  MaxDD {w.combined_report.max_drawdown:.2%}"
        )

    print("\nSensitivity -- skip_days (long_days=252):")
    for sd in [0, 10, 21, 42]:
        w = walk_forward_multi(
            prices, lambda _, s=sd: FiveAssetRotation(long_days=252, skip_days=s), folds=folds
        )
        lo_v, hi_v = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  skip={sd}d  Sharpe {w.combined_report.sharpe:.3f}"
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
    print(
        f"SPY MaxDD: -55.92%  ->  H055 MaxDD: {rep.max_drawdown:.2%}"
        f"  (improvement: {-55.92 - rep.max_drawdown * 100:.1f}pp)"
    )
    decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="055",
        name="five-asset-rotation",
        strategy_class="FiveAssetRotation",
        family="multi-asset rotation",
        sharpe=rep.sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=rep.cagr,
        max_drawdown=rep.max_drawdown,
        n_trades=rep.n_trades,
        h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results],
        decision=decision,
        notes="Cross-sectional 12-1m momentum across SPY/TLT/GLD/DBC/VNQ; cash when all negative.",
        canonical_params={"long_days": 252, "skip_days": 21},
    )


if __name__ == "__main__":
    main()
