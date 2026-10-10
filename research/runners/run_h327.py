"""H327: VolAccelerationDetector -- H086 vol-target with acceleration penalty/bonus.

When 21d vol > 1.5x 63d vol (accelerating), multiply H086 scale by 0.75 penalty.
When 21d vol < 0.75x 63d vol (decelerating), multiply by 1.15 bonus (capped at 1.0).

Run: uv run python scripts/run_h327.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies import (
    LowVolTargetPortfolio,
    MovingAverageCrossover,
    VolAccelerationDetector,
)
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    prices = spy.join(tlt, how="left")
    prices["tlt_close"] = prices["tlt_close"].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H327: VolAccelerationDetector (H086 + accel penalty/decel bonus)\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward_multi(prices, lambda _: VolAccelerationDetector(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print("\nCanonical (vol_target=0.08, accel_threshold=1.5, decel_threshold=0.75):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    wf_h086 = walk_forward_multi(prices, lambda _: LowVolTargetPortfolio(), folds=folds)
    rep86 = wf_h086.combined_report
    print(f"\nH086 baseline: Sharpe {rep86.sharpe:.3f}  MaxDD {rep86.max_drawdown:.2%}")

    print("\nSensitivity -- accel_threshold:")
    for at in [1.25, 1.5, 2.0]:
        wf_s = walk_forward_multi(
            prices, lambda _, v=at: VolAccelerationDetector(accel_threshold=v), folds=folds
        )
        rep_s = wf_s.combined_report
        lo_s, hi_s = bootstrap_sharpe_ci(wf_s.combined_returns, n_boot=1000)
        print(
            f"  accel_threshold={at}  Sharpe {rep_s.sharpe:.3f}  CI [{lo_s:.3f}, {hi_s:.3f}]  MaxDD {rep_s.max_drawdown:.2%}"
        )

    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold_result in enumerate(wf.fold_results):
        fs = float(sharpe(fold_result.net_returns))
        fold_sharpes.append(fs)
        print(
            f"  Fold {i:>2}: {fs:+.3f}  ({folds[i].test_start.date()} -- {folds[i].test_end.date()})"
        )

    folds_spy = make_folds(spy, train_years=5, test_years=2)
    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_spy)
    corr = float(
        wf.combined_returns.corr(wf_h001.combined_returns.reindex(wf.combined_returns.index))
    )
    print(f"\nH001 correlation: {corr:+.4f}")

    decision = "KEEP" if lo > 0 and rep.sharpe > 0.51 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="327",
        name="vol-acceleration-detector",
        strategy_class="VolAccelerationDetector",
        family="vol control",
        sharpe=rep.sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=rep.cagr,
        max_drawdown=rep.max_drawdown,
        n_trades=rep.n_trades,
        h001_corr=corr,
        fold_sharpes=fold_sharpes,
        decision=decision,
        notes=(
            "H086 with acceleration-aware penalty: scale*0.75 when 21d>1.5x63d, "
            "scale*1.15 (capped at 1.0) when 21d<0.75x63d."
        ),
        canonical_params={
            "vol_target": 0.08,
            "accel_threshold": 1.5,
            "decel_threshold": 0.75,
            "accel_penalty": 0.75,
            "decel_bonus": 1.15,
        },
    )


if __name__ == "__main__":
    main()
