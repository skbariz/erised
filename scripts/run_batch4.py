"""Batch 4 walk-forward runner: H023-H027 (seasonality & calendar effects).

Run: uv run python scripts/run_batch4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import (
    JanuaryBarometer,
    MovingAverageCrossover,
    QuarterEndDrift,
    SeasonalityTrendCombo,
    SellInMay,
    TurnOfMonth,
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
    print(
        f"  Sharpe {rep.sharpe:.2f}  CI [{lo:.2f}, {hi:.2f}]  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}  Trades {rep.n_trades}"
    )
    print(f"  SPY benchmark Sharpe: {bench_sh:.2f}")
    fold_sharpes = [sharpe(r.net_returns) for r in wf.fold_results]
    print(f"  Per-fold: {[f'{s:.2f}' for s in fold_sharpes]}")
    return wf, rep, lo, hi


def main():
    prices = load_research("SPY")
    print(
        f"SPY research: {len(prices)} rows {prices.index.min().date()} to {prices.index.max().date()}"
    )

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"Folds: {len(folds)}")

    # H001 reference
    print(section("H001 baseline (reference)"))
    wf_h001 = walk_forward(
        prices, strategy_factory=lambda _: MovingAverageCrossover(50, 200), folds=folds
    )
    print(f"  H001 MA(50,200) Sharpe: {wf_h001.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ H023
    print(section("H023: SellInMay  canonical (Nov-Apr long)"))
    wf23, rep23, lo23, hi23 = run(prices, folds, SellInMay, "canonical")
    print("  Sensitivity (favorable_months):")
    for fm in [(11, 12, 1, 2, 3, 4), (10, 11, 12, 1, 2, 3, 4), (11, 12, 1, 2, 3)]:
        wf = walk_forward(prices, lambda _, m=fm: SellInMay(favorable_months=m), folds=folds)
        print(f"    months={list(fm)}  Sharpe {wf.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ H024
    print(section("H024: TurnOfMonth  canonical (3 days each end)"))
    wf24, rep24, lo24, hi24 = run(prices, folds, TurnOfMonth, "canonical")
    print("  Sensitivity (n_days):")
    for nd in [2, 3, 5]:
        wf = walk_forward(prices, lambda _, n=nd: TurnOfMonth(n_days=n), folds=folds)
        print(f"    n_days={nd}  Sharpe {wf.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ H025
    print(section("H025: JanuaryBarometer  canonical"))
    wf25, rep25, lo25, hi25 = run(prices, folds, JanuaryBarometer, "canonical")

    # ------------------------------------------------------------------ H026
    print(section("H026: QuarterEndDrift  canonical (5 days)"))
    wf26, rep26, lo26, hi26 = run(prices, folds, QuarterEndDrift, "canonical")
    print("  Sensitivity (n_days):")
    for nd in [3, 5, 10]:
        wf = walk_forward(prices, lambda _, n=nd: QuarterEndDrift(n_days=n), folds=folds)
        print(f"    n_days={nd}  Sharpe {wf.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ H027
    print(section("H027: SeasonalityTrendCombo  canonical (MA 50/200, Nov-Apr)"))
    wf27, rep27, lo27, hi27 = run(prices, folds, SeasonalityTrendCombo, "canonical")
    print("  Sensitivity (MA params):")
    for f, s in [(20, 100), (50, 200), (50, 150)]:
        wf = walk_forward(prices, lambda _, ff=f, ss=s: SeasonalityTrendCombo(ff, ss), folds=folds)
        print(f"    MA({f},{s})  Sharpe {wf.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ Correlation table
    print(section("CORRELATION WITH H001 (net returns, same folds)"))
    strategies = [("H023", wf23), ("H024", wf24), ("H025", wf25), ("H026", wf26), ("H027", wf27)]
    h001_ret = wf_h001.combined_returns
    print(f"  {'Strategy':<8} {'Sharpe':>8} {'vs H001':>10} {'Decision':>10}")
    for name, wf in strategies:
        corr = wf.combined_returns.corr(h001_ret)
        sh = wf.combined_report.sharpe
        lo_v = bootstrap_sharpe_ci(wf.combined_returns, n_boot=1000)[0]
        keep = sh > 0.51 and lo_v > 0
        decision = "KEEP" if keep else "kill"
        print(f"  {name:<8} {sh:>8.2f} {corr:>+10.3f}  {decision}")


if __name__ == "__main__":
    main()
