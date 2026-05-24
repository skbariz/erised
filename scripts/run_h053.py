"""H053: CreditSpreadRegime -- standalone walk-forward runner.

Run: uv run python scripts/run_h053.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import CreditSpreadRegime, MovingAverageCrossover
from quant.validation.walk_forward import make_folds, walk_forward

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    hyg = load_research("HYG")[["adj_close"]].rename(columns={"adj_close": "hyg_close"})
    lqd = load_research("LQD")[["adj_close"]].rename(columns={"adj_close": "lqd_close"})
    prices = spy.join(hyg, how="left").join(lqd, how="left")
    prices["hyg_close"] = prices["hyg_close"].ffill()
    prices["lqd_close"] = prices["lqd_close"].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H053: CreditSpreadRegime (canonical momentum_days=63d)\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward(prices, lambda _: CreditSpreadRegime(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print("\nCanonical (momentum_days=63d):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    print("\nSensitivity -- momentum_days:")
    for md in [21, 42, 63, 126, 189]:
        w = walk_forward(prices, lambda _, m=md: CreditSpreadRegime(momentum_days=m), folds=folds)
        lo_v, hi_v = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  {md:>4}d  Sharpe {w.combined_report.sharpe:.3f}"
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
        f"SPY MaxDD: -55.92%  ->  H053 MaxDD: {rep.max_drawdown:.2%}"
        f"  (improvement: {-55.92 - rep.max_drawdown * 100:.1f}pp)"
    )
    decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="053",
        name="credit-spread-regime",
        strategy_class="CreditSpreadRegime",
        family="cross-asset macro",
        sharpe=rep.sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=rep.cagr,
        max_drawdown=rep.max_drawdown,
        n_trades=rep.n_trades,
        h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results],
        decision=decision,
        notes="Long SPY when HYG outperforms LQD (healthy credit appetite); flat otherwise.",
        canonical_params={"momentum_days": 63},
    )


if __name__ == "__main__":
    main()
