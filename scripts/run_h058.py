"""H058: InvertedCurveTreasuryLadder -- standalone walk-forward runner.

Run: uv run python scripts/run_h058.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import InvertedCurveTreasuryLadder, MovingAverageCrossover
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    shy = load_research("SHY")[["adj_close"]].rename(columns={"adj_close": "shy_close"})
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    ief = load_research("IEF")[["adj_close"]].rename(columns={"adj_close": "ief_close"})
    prices = spy.join(shy, how="left").join(tlt, how="left").join(ief, how="left")
    prices["shy_close"] = prices["shy_close"].ffill()
    prices["tlt_close"] = prices["tlt_close"].ffill()
    prices["ief_close"] = prices["ief_close"].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H058: InvertedCurveTreasuryLadder (canonical mom_days=63, sustain_days=21)\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward_multi(prices, lambda _: InvertedCurveTreasuryLadder(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print("\nCanonical (mom_days=63, sustain_days=21):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    print("\nSensitivity -- sustain_days (mom_days=63):")
    for sd in [5, 10, 21, 42, 63]:
        w = walk_forward_multi(
            prices,
            lambda _, s=sd: InvertedCurveTreasuryLadder(mom_days=63, sustain_days=s),
            folds=folds,
        )
        lo_v, hi_v = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  sustain={sd}d  Sharpe {w.combined_report.sharpe:.3f}"
            f"  CI [{lo_v:.3f}, {hi_v:.3f}]  MaxDD {w.combined_report.max_drawdown:.2%}"
        )

    print("\nSensitivity -- mom_days (sustain_days=21):")
    for md in [21, 42, 63, 126]:
        w = walk_forward_multi(
            prices,
            lambda _, m=md: InvertedCurveTreasuryLadder(mom_days=m, sustain_days=21),
            folds=folds,
        )
        lo_v, hi_v = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  mom={md}d  Sharpe {w.combined_report.sharpe:.3f}"
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
        f"SPY MaxDD: -55.92%  ->  H058 MaxDD: {rep.max_drawdown:.2%}"
        f"  (improvement: {-55.92 - rep.max_drawdown * 100:.1f}pp)"
    )
    decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="058",
        name="inverted-curve-treasury-ladder",
        strategy_class="InvertedCurveTreasuryLadder",
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
        notes="75% TLT + 25% IEF on sustained inversion (21d+); noise filter vs raw H056.",
        canonical_params={"mom_days": 63, "sustain_days": 21},
    )


if __name__ == "__main__":
    main()
