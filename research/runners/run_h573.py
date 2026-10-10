"""H573: QQQIWMTLTRiskParity -- three-asset QQQ/IWM/TLT risk parity.

Growth (QQQ) + small-cap (IWM) + bonds (TLT) inverse-vol risk parity.
Gracefully degrades as assets become available. Folds 0-2 mostly cash.

Run: uv run python scripts/run_h573.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_summary import write_summary

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies.base import MovingAverageCrossover, QQQIWMTLTRiskParity
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    qqq = load_research("QQQ")[["adj_close"]].rename(columns={"adj_close": "qqq_close"})
    iwm = load_research("IWM")[["adj_close"]].rename(columns={"adj_close": "iwm_close"})
    prices = spy.join(tlt, how="left").join(qqq, how="left").join(iwm, how="left")
    for col in ["tlt_close", "qqq_close", "iwm_close"]:
        prices[col] = prices[col].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H573: QQQIWMTLTRiskParity\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")

    wf = walk_forward_multi(prices, lambda _: QQQIWMTLTRiskParity(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print(f"\nCanonical: Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}")

    print("\nSensitivity -- vol_lookback (days):")
    for vl in [20, 40, 60, 120]:
        w = walk_forward_multi(prices, lambda _, v=vl: QQQIWMTLTRiskParity(vol_lookback=v), folds=folds)
        lo_v, hi_v = bootstrap_sharpe_ci(w.combined_returns, n_boot=1000)
        print(f"  {vl:>4}d  Sharpe {w.combined_report.sharpe:.3f}  CI [{lo_v:.3f}, {hi_v:.3f}]  MaxDD {w.combined_report.max_drawdown:.2%}")

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
        hypothesis_id="573",
        name="qqq-iwm-tlt-risk-parity",
        strategy_class="QQQIWMTLTRiskParity",
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
        notes="QQQ/IWM/TLT 3-asset risk parity. Both equity ETFs start after 2000; folds 0-2 mostly cash.",
        canonical_params={"vol_lookback": 60},
    )


if __name__ == "__main__":
    main()
