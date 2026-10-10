"""H571: QQQTLTRiskParity -- QQQ/TLT inverse-vol risk parity.

Same mechanism as H021 but using Nasdaq-100 (QQQ) instead of SPY for the
equity leg. QQQ available from 1999-03; folds 0-1 mostly cash. Folds 2-11
contribute full QQQ/TLT parity signal.

Run: uv run python scripts/run_h571.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies.base import MovingAverageCrossover, QQQTLTRiskParity
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    qqq = load_research("QQQ")[["adj_close"]].rename(columns={"adj_close": "qqq_close"})
    prices = spy.join(tlt, how="left").join(qqq, how="left")
    prices["tlt_close"] = prices["tlt_close"].ffill()
    prices["qqq_close"] = prices["qqq_close"].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H571: QQQTLTRiskParity\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    wf = walk_forward_multi(prices, lambda _: QQQTLTRiskParity(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print(f"\nCanonical (vol_lookback=60d):")
    print(f"  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}")

    print("\nSensitivity -- vol_lookback (days):")
    for vl in [20, 40, 60, 120, 252]:
        w = walk_forward_multi(prices, lambda _, v=vl: QQQTLTRiskParity(vol_lookback=v), folds=folds)
        lo_v, hi_v = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(
            f"  {vl:>4}d  Sharpe {w.combined_report.sharpe:.3f}  CI [{lo_v:.3f}, {hi_v:.3f}]  MaxDD {w.combined_report.max_drawdown:.2%}"
        )

    print("\nPer-fold Sharpe:")
    for i, fold_result in enumerate(wf.fold_results):
        print(
            f"  Fold {i:>2}: {sharpe(fold_result.net_returns):+.3f}  ({folds[i].test_start.date()} -- {folds[i].test_end.date()})"
        )

    folds_spy = make_folds(spy, train_years=5, test_years=2)
    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_spy)
    corr = wf.combined_returns.corr(wf_h001.combined_returns.reindex(wf.combined_returns.index))
    print(f"\nH001 correlation: {corr:+.4f}")

    decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="571",
        name="qqq-tlt-risk-parity",
        strategy_class="QQQTLTRiskParity",
        family="Set8 Batch2 AA",
        sharpe=rep.sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=rep.cagr,
        max_drawdown=rep.max_drawdown,
        n_trades=rep.n_trades,
        h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results],
        decision=decision,
        notes="QQQ/TLT risk parity. QQQ starts 1999-03; folds 0-1 mostly cash. Folds 2-11 active.",
        canonical_params={"vol_lookback": 60},
    )


if __name__ == "__main__":
    main()
