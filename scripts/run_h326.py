"""H326: DualFrequencyVolTarget -- vol-target using max(21d, 63d) portfolio vol.

More conservative than H086: uses the higher of short-term and long-term realized
vol as the denominator, so a short-term vol spike forces more de-risking.

Run: uv run python scripts/run_h326.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import (
    DualFrequencyVolTarget,
    LowVolTargetPortfolio,
    MovingAverageCrossover,
)
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    prices = spy.join(tlt, how="left")
    prices["tlt_close"] = prices["tlt_close"].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H326: DualFrequencyVolTarget (max(21d, 63d) vol-target)\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward_multi(prices, lambda _: DualFrequencyVolTarget(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print("\nCanonical (vol_target=0.08, short=21, long=63):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    wf_h086 = walk_forward_multi(prices, lambda _: LowVolTargetPortfolio(), folds=folds)
    rep86 = wf_h086.combined_report
    print(f"\nH086 baseline: Sharpe {rep86.sharpe:.3f}  MaxDD {rep86.max_drawdown:.2%}")

    print("\nSensitivity -- vol_target:")
    for vt in [0.06, 0.08, 0.10]:
        wf_s = walk_forward_multi(
            prices, lambda _, v=vt: DualFrequencyVolTarget(vol_target=v), folds=folds
        )
        rep_s = wf_s.combined_report
        lo_s, hi_s = bootstrap_sharpe_ci(wf_s.combined_returns, n_boot=1000)
        print(
            f"  vol_target={vt}  Sharpe {rep_s.sharpe:.3f}  CI [{lo_s:.3f}, {hi_s:.3f}]  MaxDD {rep_s.max_drawdown:.2%}"
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
        hypothesis_id="326",
        name="dual-frequency-vol-target",
        strategy_class="DualFrequencyVolTarget",
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
            "H086 variant using max(21d, 63d) realized vol in the denominator. "
            "More conservative than H086 during short-term vol spikes."
        ),
        canonical_params={"vol_target": 0.08, "short_window": 21, "long_window": 63},
    )


if __name__ == "__main__":
    main()
