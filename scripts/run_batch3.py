"""Batch 3 walk-forward runner: H018-H022 (multi-asset rotation).

Run: uv run python scripts/run_batch3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import (
    DefensiveEquityHedge,
    MovingAverageCrossover,
    RiskParity,
    SPYGLDRotation,
    SPYTLTDualMomentum,
    ThreeAssetMomentum,
)
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def section(title):
    return f"\n{SEP}\n  {title}\n{SEP}"


def run_multi(prices, folds, factory, label):
    wf = walk_forward_multi(prices, strategy_factory=lambda _: factory(), folds=folds)
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
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    gld = load_research("GLD")[["adj_close"]].rename(columns={"adj_close": "gld_close"})

    # Multi-asset prices DataFrame: join on shared dates
    prices = spy.join(tlt, how="left").join(gld, how="left")
    prices["tlt_close"] = prices["tlt_close"].ffill()
    prices["gld_close"] = prices["gld_close"].ffill()

    print(
        f"Multi-asset prices: {len(prices)} rows {prices.index.min().date()} to {prices.index.max().date()}"
    )
    print(f"  Columns: {list(prices.columns)}")

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"Folds: {len(folds)}")

    # H001 reference (single-asset)
    print(section("H001 baseline (reference)"))
    wf_h001 = walk_forward(
        spy,
        strategy_factory=lambda _: MovingAverageCrossover(50, 200),
        folds=make_folds(spy, train_years=5, test_years=2),
    )
    print(f"  H001 MA(50,200) Sharpe: {wf_h001.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ H018
    print(section("H018: SPYTLTDualMomentum  canonical (12m-1m lookback)"))
    wf18, rep18, lo18, hi18 = run_multi(prices, folds, SPYTLTDualMomentum, "canonical")
    print("  1-D sensitivity (lookback_months):")
    for lm in [6, 9, 12]:
        wf = walk_forward_multi(
            prices, lambda _, l=lm: SPYTLTDualMomentum(lookback_months=l), folds=folds
        )
        print(f"    lookback={lm}m  Sharpe {wf.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ H019
    print(section("H019: SPYGLDRotation  canonical (12m-1m lookback)"))
    wf19, rep19, lo19, hi19 = run_multi(prices, folds, SPYGLDRotation, "canonical")
    print("  1-D sensitivity (lookback_months):")
    for lm in [6, 9, 12]:
        wf = walk_forward_multi(
            prices, lambda _, l=lm: SPYGLDRotation(lookback_months=l), folds=folds
        )
        print(f"    lookback={lm}m  Sharpe {wf.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ H020
    print(section("H020: ThreeAssetMomentum  canonical (12m-1m lookback)"))
    wf20, rep20, lo20, hi20 = run_multi(prices, folds, ThreeAssetMomentum, "canonical")
    print("  1-D sensitivity (lookback_months):")
    for lm in [6, 9, 12]:
        wf = walk_forward_multi(
            prices, lambda _, l=lm: ThreeAssetMomentum(lookback_months=l), folds=folds
        )
        print(f"    lookback={lm}m  Sharpe {wf.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ H021
    print(section("H021: RiskParity  canonical (vol_lookback=60d)"))
    wf21, rep21, lo21, hi21 = run_multi(prices, folds, RiskParity, "canonical")
    print("  1-D sensitivity (vol_lookback):")
    for vl in [20, 60, 120]:
        wf = walk_forward_multi(prices, lambda _, v=vl: RiskParity(vol_lookback=v), folds=folds)
        print(f"    vol_lookback={vl}d  Sharpe {wf.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ H022
    print(section("H022: DefensiveEquityHedge  canonical (fast=50, slow=200)"))
    wf22, rep22, lo22, hi22 = run_multi(prices, folds, DefensiveEquityHedge, "canonical")
    print("  1-D sensitivity (fast/slow):")
    for f, s in [(20, 100), (50, 200), (50, 150)]:
        wf = walk_forward_multi(
            prices, lambda _, ff=f, ss=s: DefensiveEquityHedge(ff, ss), folds=folds
        )
        print(f"    MA({f},{s})  Sharpe {wf.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ Correlation table
    print(section("CORRELATION WITH H001 (net returns, same folds)"))
    strategies = [("H018", wf18), ("H019", wf19), ("H020", wf20), ("H021", wf21), ("H022", wf22)]
    # Align H001 returns to multi-asset fold dates
    h001_ret = wf_h001.combined_returns
    print(f"  {'Strategy':<8} {'Sharpe':>8} {'vs H001':>10} {'Decision':>10}")
    for name, wf in strategies:
        corr = wf.combined_returns.corr(h001_ret.reindex(wf.combined_returns.index))
        sh = wf.combined_report.sharpe
        lo_v = bootstrap_sharpe_ci(wf.combined_returns, n_boot=1000)[0]
        keep = sh > 0.51 and lo_v > 0
        decision = "KEEP" if keep else "kill"
        print(f"  {name:<8} {sh:>8.2f} {corr:>+10.3f}  {decision}")


if __name__ == "__main__":
    main()
