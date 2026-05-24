"""H065: EqualWeightDefensiveFactor -- standalone walk-forward runner.

Run: uv run python scripts/run_h065.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import EqualWeightDefensiveFactor, MovingAverageCrossover
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    xlp = load_research("XLP")[["adj_close"]].rename(columns={"adj_close": "xlp_close"})
    xlu = load_research("XLU")[["adj_close"]].rename(columns={"adj_close": "xlu_close"})
    ive = load_research("IVE")[["adj_close"]].rename(columns={"adj_close": "ive_close"})
    prices = spy.join(xlp, how="left").join(xlu, how="left").join(ive, how="left")
    prices["xlp_close"] = prices["xlp_close"].ffill()
    prices["xlu_close"] = prices["xlu_close"].ffill()
    prices["ive_close"] = prices["ive_close"].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H065: EqualWeightDefensiveFactor (static 25/25/25/25)\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward_multi(prices, lambda _: EqualWeightDefensiveFactor(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print("\nCanonical (static equal-weight, no parameters):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

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
        f"SPY MaxDD: -55.92%  ->  H065 MaxDD: {rep.max_drawdown:.2%}"
        f"  (improvement: {-55.92 - rep.max_drawdown * 100:.1f}pp)"
    )
    decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="065",
        name="equal-weight-defensive-factor",
        strategy_class="EqualWeightDefensiveFactor",
        family="equity factor",
        sharpe=rep.sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=rep.cagr,
        max_drawdown=rep.max_drawdown,
        n_trades=rep.n_trades,
        h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results],
        decision=decision,
        notes="Static 25/25/25/25 SPY+XLP+XLU+IVE. No timing. Pure factor quality tilt.",
        canonical_params={},
    )


if __name__ == "__main__":
    main()
