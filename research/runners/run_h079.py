"""H079: CrisisProtocolRiskParity -- standalone walk-forward runner.

Run: uv run python scripts/run_h079.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies.base import CrisisProtocolRiskParity, MovingAverageCrossover, RiskParity
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})
    prices = spy.join(tlt, how="left").join(vix, how="left")
    prices["tlt_close"] = prices["tlt_close"].ffill()
    prices["vix_close"] = prices["vix_close"].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(
        f"H079: CrisisProtocolRiskParity (canonical dd_window=20, crisis_dd=-10%, vix_spike=35)\n{SEP}"
    )
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward_multi(prices, lambda _: CrisisProtocolRiskParity(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print("\nCanonical (dd_window=20, crisis_dd=-10%, vix_spike=35, crisis_tlt=90%):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    wf_h021 = walk_forward_multi(prices, lambda _: RiskParity(60), folds=folds)
    rep_h021 = wf_h021.combined_report
    print(f"\nH021 baseline: Sharpe {rep_h021.sharpe:.3f}  MaxDD {rep_h021.max_drawdown:.2%}")

    print("\nSensitivity -- crisis_tlt:")
    for ct in [0.70, 0.80, 0.90, 1.00]:
        w = walk_forward_multi(
            prices,
            lambda _, c=ct: CrisisProtocolRiskParity(crisis_tlt=c),
            folds=folds,
        )
        lo_v, hi_v = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  crisis_tlt={ct:.0%}  Sharpe {w.combined_report.sharpe:.3f}"
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
        hypothesis_id="079",
        name="crisis-protocol-risk-parity",
        strategy_class="CrisisProtocolRiskParity",
        family="multi-asset",
        sharpe=rep.sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=rep.cagr,
        max_drawdown=rep.max_drawdown,
        n_trades=rep.n_trades,
        h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results],
        decision=decision,
        notes="H021 with hysteresis-based emergency override: 90/10 TLT/SPY when SPY drawdown<-10% AND VIX>35.",
        canonical_params={
            "vol_lookback": 60,
            "drawdown_window": 20,
            "crisis_dd": -0.10,
            "vix_spike": 35.0,
            "crisis_tlt": 0.90,
        },
    )


if __name__ == "__main__":
    main()
