"""H340: GrandCrisisProtocol -- union of 4 crisis signals, tiered SPY/TLT response.

Run: uv run python scripts/run_h340.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies.base import GrandCrisisProtocol, MovingAverageCrossover, RiskParity
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})
    hyg = load_research("HYG")[["adj_close"]].rename(columns={"adj_close": "hyg_close"})
    lqd = load_research("LQD")[["adj_close"]].rename(columns={"adj_close": "lqd_close"})
    shy = load_research("SHY")[["adj_close"]].rename(columns={"adj_close": "shy_close"})
    prices = (
        spy.join(tlt, how="left")
        .join(vix, how="left")
        .join(hyg, how="left")
        .join(lqd, how="left")
        .join(shy, how="left")
    )
    for col in ["tlt_close", "vix_close", "hyg_close", "lqd_close", "shy_close"]:
        prices[col] = prices[col].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H340: GrandCrisisProtocol (4 crisis signals, 20/80 or 40/60 tiered TLT)\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward_multi(prices, lambda _: GrandCrisisProtocol(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print("\nCanonical (dd_threshold=0.15, vix_threshold=30, stress_high=0.67, vrp_severe=-0.005):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    wf_h021 = walk_forward_multi(prices, lambda _: RiskParity(), folds=folds)
    rep21 = wf_h021.combined_report
    print(f"\nH021 baseline: Sharpe {rep21.sharpe:.3f}  MaxDD {rep21.max_drawdown:.2%}")

    print("\nSensitivity -- stress_high (threshold for signal 3):")
    for sh in [0.55, 0.67, 0.80]:
        wf_s = walk_forward_multi(
            prices,
            lambda _, v=sh: GrandCrisisProtocol(stress_high=v),
            folds=folds,
        )
        rep_s = wf_s.combined_report
        lo_s, hi_s = bootstrap_sharpe_ci(wf_s.combined_returns, n_boot=1000)
        print(
            f"  stress_high={sh:.2f}  Sharpe {rep_s.sharpe:.3f}  CI [{lo_s:.3f}, {hi_s:.3f}]  MaxDD {rep_s.max_drawdown:.2%}"
        )

    print("\nSensitivity -- tlt_two (max defense TLT weight):")
    for tt in [0.70, 0.80, 0.90]:
        wf_s = walk_forward_multi(
            prices,
            lambda _, v=tt: GrandCrisisProtocol(tlt_two=v, spy_two=1.0 - v),
            folds=folds,
        )
        rep_s = wf_s.combined_report
        lo_s, hi_s = bootstrap_sharpe_ci(wf_s.combined_returns, n_boot=1000)
        print(
            f"  tlt_two={tt:.2f}  Sharpe {rep_s.sharpe:.3f}  CI [{lo_s:.3f}, {hi_s:.3f}]  MaxDD {rep_s.max_drawdown:.2%}"
        )

    print("\nSensitivity -- dd_threshold:")
    for dt in [0.10, 0.15, 0.20]:
        wf_s = walk_forward_multi(
            prices,
            lambda _, v=dt: GrandCrisisProtocol(dd_threshold=v),
            folds=folds,
        )
        rep_s = wf_s.combined_report
        lo_s, hi_s = bootstrap_sharpe_ci(wf_s.combined_returns, n_boot=1000)
        print(
            f"  dd_threshold={dt:.2f}  Sharpe {rep_s.sharpe:.3f}  CI [{lo_s:.3f}, {hi_s:.3f}]  MaxDD {rep_s.max_drawdown:.2%}"
        )

    print("\nPer-fold Sharpe:")
    for i, fold_result in enumerate(wf.fold_results):
        print(
            f"  Fold {i:>2}: {sharpe(fold_result.net_returns):+.3f}  ({folds[i].test_start.date()} -- {folds[i].test_end.date()})"
        )

    folds_spy = make_folds(spy, train_years=5, test_years=2)
    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_spy)
    corr = wf.combined_returns.corr(wf_h001.combined_returns.reindex(wf.combined_returns.index))
    print(f"\nH001 correlation: {corr:+.4f}")
    decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="340",
        name="grand-crisis-protocol",
        strategy_class="GrandCrisisProtocol",
        family="vol control",
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
            "4 crisis signals (H097 corr, H079 DD+VIX, H306 stress, VRP<-0.005). "
            "2+ signals: 20% SPY / 80% TLT. "
            "1 signal: 40% SPY / 60% TLT. "
            "0 signals: H021 inverse-vol weights."
        ),
        canonical_params={
            "vol_lookback": 60,
            "corr_lookback": 60,
            "dd_threshold": 0.15,
            "vix_threshold": 30.0,
            "stress_high": 0.67,
            "vrp_severe": -0.005,
            "spy_one": 0.40,
            "tlt_one": 0.60,
            "spy_two": 0.20,
            "tlt_two": 0.80,
        },
    )


if __name__ == "__main__":
    main()
