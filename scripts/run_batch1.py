"""Batch 1 walk-forward runner: H003-H012.

Run: uv run python scripts/run_batch1.py

Evaluates all 10 strategies with the same folds as H001/H002 and writes
results to stdout. Copy numbers into research logs after review.
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, report, sharpe
from quant.strategies.base import (
    MovingAverageCrossover,
    RSIReversionStrategy,
    ShortHoldMeanReversion,
)
from quant.validation.walk_forward import make_folds, walk_forward

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
    print(f"  Sharpe {rep.sharpe:.2f}  CI [{lo:.2f}, {hi:.2f}]  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  Trades {rep.n_trades}")
    print(f"  SPY benchmark Sharpe: {bench_sh:.2f}")
    fold_sharpes = [sharpe(r.net_returns) for r in wf.fold_results]
    print(f"  Per-fold: {[f'{s:.2f}' for s in fold_sharpes]}")
    return wf, rep, lo, hi


def h001_baseline(prices, folds):
    wf = walk_forward(prices, strategy_factory=lambda _: MovingAverageCrossover(50, 200), folds=folds)
    return wf


def main():
    prices = load_research("SPY")
    print(f"SPY research: {len(prices)} rows {prices.index.min().date()} to {prices.index.max().date()}")


    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"Folds: {len(folds)}")

    print(section("H001 baseline (reference)"))
    wf_h001 = h001_baseline(prices, folds)
    print(f"  H001 MA(50,200) Sharpe: {wf_h001.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ H003
    print(section("H003: ShortHoldMeanReversion  canonical (lookback=20, thresh=2.0, hold=2)"))
    wf3, rep3, lo3, hi3 = run(prices, folds, ShortHoldMeanReversion, "canonical")
    print("  1-D sensitivity (hold_days):")
    for hd in [1, 2, 3]:
        wf = walk_forward(prices, lambda _, h=hd: ShortHoldMeanReversion(hold_days=h), folds=folds)
        print(f"    hold_days={hd}  Sharpe {wf.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ H004
    print(section("H004: RSIReversionStrategy  canonical (period=14, entry=30, exit=50, hold=5)"))
    wf4, rep4, lo4, hi4 = run(prices, folds, RSIReversionStrategy, "canonical")
    print("  1-D sensitivity (entry_rsi):")
    for ent in [25, 30, 35]:
        wf = walk_forward(prices, lambda _, e=ent: RSIReversionStrategy(entry_rsi=e), folds=folds)
        print(f"    entry_rsi={ent}  Sharpe {wf.combined_report.sharpe:.2f}")



if __name__ == "__main__":
    main()
