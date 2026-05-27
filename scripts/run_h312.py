"""H312: DefaultSpreadSPYTimer -- H021 with HYG vs LQD default spread signal.

Run: uv run python scripts/run_h312.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import DefaultSpreadSPYTimer, MovingAverageCrossover, RiskParity
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    hyg = load_research("HYG")[["adj_close"]].rename(columns={"adj_close": "hyg_close"})
    lqd = load_research("LQD")[["adj_close"]].rename(columns={"adj_close": "lqd_close"})
    prices = spy.join(tlt, how="left").join(hyg, how="left").join(lqd, how="left")
    for col in ["tlt_close", "hyg_close", "lqd_close"]:
        prices[col] = prices[col].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H312: DefaultSpreadSPYTimer (HYG vs LQD 63d default spread on H021)\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward_multi(prices, lambda _: DefaultSpreadSPYTimer(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print(f"\nCanonical (credit_lookback=63, spy_scale=0.60):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    wf_h021 = walk_forward_multi(prices, lambda _: RiskParity(60), folds=folds)
    print(f"\nH021 baseline: Sharpe {wf_h021.combined_report.sharpe:.3f}  MaxDD {wf_h021.combined_report.max_drawdown:.2%}")

    print("\nSensitivity -- credit_lookback:")
    for lb in [42, 63, 126]:
        w = walk_forward_multi(prices, lambda _, l=lb: DefaultSpreadSPYTimer(credit_lookback=l), folds=folds)
        lo_s, hi_s = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(f"  credit_lookback={lb}  Sharpe {w.combined_report.sharpe:.3f}  CI [{lo_s:.3f}, {hi_s:.3f}]  MaxDD {w.combined_report.max_drawdown:.2%}")

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
        hypothesis_id="312",
        name="default-spread-spy-timer",
        strategy_class="DefaultSpreadSPYTimer",
        family="macro timing",
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
            "H021 with HYG vs LQD 63d return as default spread proxy. "
            "Credit risk-off (LQD > HYG) → SPY weight × 0.60; freed weight to TLT. "
            "Pre-2007 (no HYG): pure H021. Goyal-Welch (2008): default spread predicts equity premium."
        ),
        canonical_params={"credit_lookback": 63, "spy_scale": 0.60},
    )


if __name__ == "__main__":
    main()
