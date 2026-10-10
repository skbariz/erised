"""Batch 5 walk-forward runner: H028-H032 (VIX-based strategies).

Run: uv run python scripts/run_batch5.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies.base import (
    MovingAverageCrossover,
    RealizedVsImpliedVol,
    VIXMeanReversion,
    VIXRegimeFilter,
    VIXSpikeEntry,
    VIXTrendSignal,
)
from erised.validation.walk_forward import make_folds, walk_forward

SEP = "=" * 64


def section(title):
    return f"\n{title}\n{'-' * len(title)}"


def run(prices, folds, cls, label, **kwargs):
    wf = walk_forward(prices, lambda _: cls(**kwargs), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print(
        f"  [{label}]  Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]"
        f"  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  N_trades {rep.n_trades}"
    )
    return wf, rep, lo, hi


def main():
    spy = load_research("SPY")
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})
    prices = spy.join(vix, how="left")
    prices["vix_close"] = prices["vix_close"].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"Batch 5: H028-H032 — VIX-Based Strategies\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} — {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    print(section("H001 baseline (reference)"))
    wf_h001 = walk_forward(
        spy,
        lambda _: MovingAverageCrossover(50, 200),
        folds=make_folds(spy, train_years=5, test_years=2),
    )
    lo_h001, hi_h001 = bootstrap_sharpe_ci(wf_h001.combined_returns, n_boot=5000)
    print(
        f"  [H001]  Sharpe {wf_h001.combined_report.sharpe:.3f}  CI [{lo_h001:.3f}, {hi_h001:.3f}]"
        f"  CAGR {wf_h001.combined_report.cagr:.2%}  MaxDD {wf_h001.combined_report.max_drawdown:.2%}"
    )

    print(section("H028: VIXSpikeEntry  canonical (vix_threshold=30, hold_days=20)"))
    wf28, rep28, lo28, hi28 = run(prices, folds, VIXSpikeEntry, "canonical")
    print("  Sensitivity (vix_threshold):")
    for vt in [25, 30, 40]:
        w = walk_forward(prices, lambda _, v=vt: VIXSpikeEntry(vix_threshold=v), folds=folds)
        print(f"    vix_threshold={vt}  Sharpe {w.combined_report.sharpe:.3f}")
    print("  Sensitivity (hold_days):")
    for hd in [10, 20, 30]:
        w = walk_forward(prices, lambda _, h=hd: VIXSpikeEntry(hold_days=h), folds=folds)
        print(f"    hold_days={hd}  Sharpe {w.combined_report.sharpe:.3f}")

    print(section("H029: VIXRegimeFilter  canonical (vix_threshold=20)"))
    wf29, rep29, lo29, hi29 = run(prices, folds, VIXRegimeFilter, "canonical")
    print("  Sensitivity (vix_threshold):")
    for vt in [15, 20, 25]:
        w = walk_forward(prices, lambda _, v=vt: VIXRegimeFilter(vix_threshold=v), folds=folds)
        print(f"    vix_threshold={vt}  Sharpe {w.combined_report.sharpe:.3f}")

    print(section("H030: VIXTrendSignal  canonical (ma_period=20)"))
    wf30, rep30, lo30, hi30 = run(prices, folds, VIXTrendSignal, "canonical")
    print("  Sensitivity (ma_period):")
    for mp in [10, 20, 60]:
        w = walk_forward(prices, lambda _, m=mp: VIXTrendSignal(ma_period=m), folds=folds)
        print(f"    ma_period={mp}  Sharpe {w.combined_report.sharpe:.3f}")

    print(section("H031: RealizedVsImpliedVol  canonical (realized_window=20)"))
    wf31, rep31, lo31, hi31 = run(prices, folds, RealizedVsImpliedVol, "canonical")
    print("  Sensitivity (realized_window):")
    for rw in [10, 20, 60]:
        w = walk_forward(
            prices, lambda _, r=rw: RealizedVsImpliedVol(realized_window=r), folds=folds
        )
        print(f"    realized_window={rw}  Sharpe {w.combined_report.sharpe:.3f}")

    print(section("H032: VIXMeanReversion  canonical (lookback=60, n_sigma=2.0)"))
    wf32, rep32, lo32, hi32 = run(prices, folds, VIXMeanReversion, "canonical")
    print("  Sensitivity (n_sigma):")
    for ns in [1.5, 2.0, 2.5]:
        w = walk_forward(prices, lambda _, n=ns: VIXMeanReversion(n_sigma=n), folds=folds)
        print(f"    n_sigma={ns}  Sharpe {w.combined_report.sharpe:.3f}")

    print(section("CORRELATION WITH H001 (net returns, same folds)"))
    strategies = [("H028", wf28), ("H029", wf29), ("H030", wf30), ("H031", wf31), ("H032", wf32)]
    h001_ret = wf_h001.combined_returns
    for name, wf in strategies:
        corr = wf.combined_returns.corr(h001_ret.reindex(wf.combined_returns.index))
        print(f"  {name}: {corr:+.4f}")

    print(section("DECISION SUMMARY"))
    results = [
        ("H028", rep28, lo28, hi28),
        ("H029", rep29, lo29, hi29),
        ("H030", rep30, lo30, hi30),
        ("H031", rep31, lo31, hi31),
        ("H032", rep32, lo32, hi32),
    ]
    for name, rep, lo, hi in results:
        decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"
        print(f"  {name}: Sharpe {rep.sharpe:.3f} CI [{lo:.3f}, {hi:.3f}]  -> {decision}")

    print(section("PER-FOLD SHARPE"))
    for name, wf in strategies:
        fold_sharpes = [f"{sharpe(fr.net_returns):+.2f}" for fr in wf.fold_results]
        print(f"  {name}: {' | '.join(fold_sharpes)}")


if __name__ == "__main__":
    main()
