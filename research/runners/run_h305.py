"""H305: VRPSeasonalHybrid -- H078 seasonal + VRP overlay in Nov-Apr.

Run: uv run python scripts/run_h305.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies.base import MovingAverageCrossover, SeasonalRiskParity, VRPSeasonalHybrid
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
    print(f"H305: VRPSeasonalHybrid (H078 + VRP overlay in Nov-Apr)\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward_multi(prices, lambda _: VRPSeasonalHybrid(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print(f"\nCanonical (unfavorable_scale=0.75):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    wf_h078 = walk_forward_multi(prices, lambda _: SeasonalRiskParity(), folds=folds)
    print(f"\nH078 baseline: Sharpe {wf_h078.combined_report.sharpe:.3f}  MaxDD {wf_h078.combined_report.max_drawdown:.2%}")

    print("\nSensitivity -- unfavorable_scale:")
    for us in [0.60, 0.75, 0.90]:
        w = walk_forward_multi(prices, lambda _, u=us: VRPSeasonalHybrid(unfavorable_scale=u), folds=folds)
        lo_s, hi_s = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(f"  unfavorable_scale={us}  Sharpe {w.combined_report.sharpe:.3f}  CI [{lo_s:.3f}, {hi_s:.3f}]  MaxDD {w.combined_report.max_drawdown:.2%}")

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
        hypothesis_id="305",
        name="vrp-seasonal-hybrid",
        strategy_class="VRPSeasonalHybrid",
        family="VRP timing",
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
            "H078 seasonal framework + VRP overlay in Nov-Apr only. "
            "May-Oct: 100% TLT (unchanged). Nov-Apr + VRP>0: H021 full; + VRP<=0: H021 × 0.75. "
            "Combines zero-corr seasonal anchor with VRP timing during active season."
        ),
        canonical_params={"unfavorable_scale": 0.75, "vol_lookback": 60},
    )


if __name__ == "__main__":
    main()
