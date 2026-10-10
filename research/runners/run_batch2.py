"""Batch 2 walk-forward runner: H013-H017 (long/short structures).

Run: uv run python scripts/run_batch2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies import (
    DualMALongShort,
    LongShortMACrossover,
    LongShortTSMomentum,
    MovingAverageCrossover,
    TrendVIXGate,
    VolScaledTrend,
)
from erised.validation.walk_forward import make_folds, walk_forward

SEP = "=" * 64


def section(title):
    return f"\n{SEP}\n  {title}\n{SEP}"


def run(prices, folds, factory, label):
    wf = walk_forward(prices, strategy_factory=lambda _: factory(), folds=folds)
    rep = wf.combined_report
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    bench = pd.concat([r.bench_returns for r in wf.fold_results]).sort_index()
    bench_sh = sharpe(bench)
    print(f"\n  {label}")
    print(
        f"  Sharpe {rep.sharpe:.2f}  CI [{lo:.2f}, {hi:.2f}]  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  Trades {rep.n_trades}"
    )
    print(f"  SPY benchmark Sharpe: {bench_sh:.2f}")
    fold_sharpes = [sharpe(r.net_returns) for r in wf.fold_results]
    print(f"  Per-fold: {[f'{s:.2f}' for s in fold_sharpes]}")
    return wf, rep, lo, hi


def h001_baseline(prices, folds):
    wf = walk_forward(
        prices, strategy_factory=lambda _: MovingAverageCrossover(50, 200), folds=folds
    )
    return wf


def main():
    prices = load_research("SPY")
    print(
        f"SPY research: {len(prices)} rows {prices.index.min().date()} to {prices.index.max().date()}"
    )

    # VIX for H017
    vix_path = Path(__file__).resolve().parents[2] / "data" / "VIX_research.parquet"
    vix = pd.read_parquet(vix_path)[["adj_close"]].rename(columns={"adj_close": "vix_close"})
    prices_vix = prices.join(vix, how="left")
    prices_vix["vix_close"] = prices_vix["vix_close"].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"Folds: {len(folds)}")

    print(section("H001 baseline (reference)"))
    wf_h001 = h001_baseline(prices, folds)
    print(f"  H001 MA(50,200) Sharpe: {wf_h001.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ H013
    print(section("H013: LongShortMACrossover  canonical (fast=50, slow=200)"))
    wf13, rep13, lo13, hi13 = run(prices, folds, LongShortMACrossover, "canonical")
    print("  1-D sensitivity (fast/slow):")
    from erised.strategies import LongShortMACrossover as LSMA

    for f, s in [(20, 100), (50, 200), (50, 150)]:
        wf = walk_forward(prices, lambda _, ff=f, ss=s: LSMA(ff, ss), folds=folds)
        print(f"    MA({f},{s})  Sharpe {wf.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ H014
    print(section("H014: LongShortTSMomentum  canonical (12m-1m skip)"))
    wf14, rep14, lo14, hi14 = run(prices, folds, LongShortTSMomentum, "canonical")
    print("  1-D sensitivity (long_months):")
    for lm in [6, 9, 12]:
        wf = walk_forward(prices, lambda _, l=lm: LongShortTSMomentum(long_months=l), folds=folds)
        print(f"    long_months={lm}  Sharpe {wf.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ H015
    print(section("H015: DualMALongShort  canonical (fast=20, slow=100)"))
    wf15, rep15, lo15, hi15 = run(prices, folds, DualMALongShort, "canonical")
    print("  1-D sensitivity (fast/slow):")
    for f, s in [(10, 50), (20, 100), (20, 50)]:
        wf = walk_forward(prices, lambda _, ff=f, ss=s: DualMALongShort(ff, ss), folds=folds)
        print(f"    MA({f},{s})  Sharpe {wf.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ H016
    print(section("H016: VolScaledTrend  canonical (fast=50, slow=200, target_vol=10%)"))
    wf16, rep16, lo16, hi16 = run(prices, folds, VolScaledTrend, "canonical")
    print("  1-D sensitivity (target_vol):")
    for tv in [0.08, 0.10, 0.15]:
        wf = walk_forward(prices, lambda _, t=tv: VolScaledTrend(target_vol=t), folds=folds)
        print(f"    target_vol={tv:.0%}  Sharpe {wf.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ H017
    print(section("H017: TrendVIXGate  canonical (fast=50, slow=200, vix_extreme=40)"))
    wf17, rep17, lo17, hi17 = run(prices_vix, folds, TrendVIXGate, "canonical")
    print("  1-D sensitivity (vix_extreme):")
    for ve in [30, 40, 50]:
        wf = walk_forward(prices_vix, lambda _, v=ve: TrendVIXGate(vix_extreme=v), folds=folds)
        print(f"    vix_extreme={ve}  Sharpe {wf.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ Correlation table
    print(section("CORRELATION WITH H001 (net returns, same folds)"))
    strategies = [("H013", wf13), ("H014", wf14), ("H015", wf15), ("H016", wf16), ("H017", wf17)]
    h001_ret = wf_h001.combined_returns
    print(f"  {'Strategy':<8} {'Sharpe':>8} {'vs H001':>10}")
    for name, wf in strategies:
        corr = wf.combined_returns.corr(h001_ret)
        print(f"  {name:<8} {wf.combined_report.sharpe:>8.2f} {corr:>+10.3f}")


if __name__ == "__main__":
    main()
