"""H310: FundingStressVolTarget -- H086 vol-target × funding stress compound.

Run: uv run python scripts/run_h310.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies.base import FundingStressVolTarget, LowVolTargetPortfolio, MovingAverageCrossover
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})
    hyg = load_research("HYG")[["adj_close"]].rename(columns={"adj_close": "hyg_close"})
    lqd = load_research("LQD")[["adj_close"]].rename(columns={"adj_close": "lqd_close"})
    prices = spy.join(tlt, how="left").join(vix, how="left").join(hyg, how="left").join(lqd, how="left")
    for col in ["tlt_close", "vix_close", "hyg_close", "lqd_close"]:
        prices[col] = prices[col].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H310: FundingStressVolTarget (H086 × binary funding stress)\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward_multi(prices, lambda _: FundingStressVolTarget(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print(f"\nCanonical (vol_target=8%, vix_threshold=30, stress_reduction=0.65):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    wf_h086 = walk_forward_multi(prices, lambda _: LowVolTargetPortfolio(), folds=folds)
    print(f"\nH086 baseline: Sharpe {wf_h086.combined_report.sharpe:.3f}  MaxDD {wf_h086.combined_report.max_drawdown:.2%}")

    print("\nSensitivity -- stress_reduction:")
    for sr in [0.50, 0.65, 0.80]:
        w = walk_forward_multi(prices, lambda _, s=sr: FundingStressVolTarget(stress_reduction=s), folds=folds)
        lo_s, hi_s = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(f"  stress_reduction={sr}  Sharpe {w.combined_report.sharpe:.3f}  CI [{lo_s:.3f}, {hi_s:.3f}]  MaxDD {w.combined_report.max_drawdown:.2%}")

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
        hypothesis_id="310",
        name="funding-stress-vol-target",
        strategy_class="FundingStressVolTarget",
        family="funding stress",
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
            "H086 vol-target (8%) compounded with binary funding stress flag. "
            "stress = VIX>30 AND HYG<LQD 63d (fallback: VIX>30 only pre-2007). "
            "final_scale = vol_scale × (0.65 if stressed else 1.0). Compound two derisking mechanisms."
        ),
        canonical_params={"vol_target": 0.08, "vix_threshold": 30.0, "stress_reduction": 0.65},
    )


if __name__ == "__main__":
    main()
