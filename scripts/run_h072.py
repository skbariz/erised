"""H072: HalloweenCreditConfirm -- standalone walk-forward runner.

Run: uv run python scripts/run_h072.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import HalloweenCreditConfirm, MovingAverageCrossover
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
    print(f"H072: HalloweenCreditConfirm (canonical mom_days=126, credit_days=60)\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward(prices, lambda _: HalloweenCreditConfirm(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print("\nCanonical (mom_days=126, credit_days=60, Nov-Apr):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    print("\nSensitivity -- credit_days (mom_days=126):")
    for cd in [30, 60, 90, 126]:
        w = walk_forward(
            prices,
            lambda _, c=cd: HalloweenCreditConfirm(mom_days=126, credit_days=c),
            folds=folds,
        )
        lo_v, hi_v = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  credit={cd}d  Sharpe {w.combined_report.sharpe:.3f}"
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
        hypothesis_id="072",
        name="halloween-credit-confirm",
        strategy_class="HalloweenCreditConfirm",
        family="calendar",
        sharpe=rep.sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=rep.cagr,
        max_drawdown=rep.max_drawdown,
        n_trades=rep.n_trades,
        h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results],
        decision=decision,
        notes="H046 (Halloween+momentum) gated by HYG>LQD credit confirmation. Flat pre-2007 (no HYG).",
        canonical_params={"mom_days": 126, "credit_days": 60},
    )


if __name__ == "__main__":
    main()
