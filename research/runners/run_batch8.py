"""Batch 8 walk-forward runner: H043-H047 (Calendar Refinements + Cross-Asset Hybrids).

Run: uv run python scripts/run_batch8.py

Calendar signals enhanced with cross-asset (TLT, GLD) and VIX filters.
Target H001 correlation: 0.10-0.40 (lower than plain calendar batch).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies.base import (
    HalloweenMomentumConfirm,
    JanuaryBarometerGoldFilter,
    MovingAverageCrossover,
    PreEarningsDrift,
    SellInMayBondConfirm,
    TurnOfMonthLowVol,
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
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    gld = load_research("GLD")[["adj_close"]].rename(columns={"adj_close": "gld_close"})
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})

    prices_tlt = spy.join(tlt, how="left")
    prices_tlt["tlt_close"] = prices_tlt["tlt_close"].ffill()

    prices_gld = spy.join(gld, how="left")
    prices_gld["gld_close"] = prices_gld["gld_close"].ffill()

    prices_vix = spy.join(vix, how="left")
    prices_vix["vix_close"] = prices_vix["vix_close"].ffill()

    folds_spy = make_folds(spy, train_years=5, test_years=2)
    folds_tlt = make_folds(prices_tlt, train_years=5, test_years=2)
    folds_gld = make_folds(prices_gld, train_years=5, test_years=2)
    folds_vix = make_folds(prices_vix, train_years=5, test_years=2)

    print(f"Batch 8: H043-H047 — Calendar Refinements + Cross-Asset Hybrids\n{SEP}")
    print(f"SPY rows: {len(spy)}")

    print(section("H001 baseline (reference)"))
    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_spy)
    lo_h001, hi_h001 = bootstrap_sharpe_ci(wf_h001.combined_returns, n_boot=5000)
    print(
        f"  [H001]  Sharpe {wf_h001.combined_report.sharpe:.3f}  CI [{lo_h001:.3f}, {hi_h001:.3f}]"
        f"  CAGR {wf_h001.combined_report.cagr:.2%}  MaxDD {wf_h001.combined_report.max_drawdown:.2%}"
    )

    print(section("H043: SellInMayBondConfirm  canonical (TLT < 200MA)"))
    wf43, rep43, lo43, hi43 = run(prices_tlt, folds_tlt, SellInMayBondConfirm, "canonical")
    print("  Sensitivity (tlt_ma_period):")
    for ma in [50, 100, 150, 200]:
        w = walk_forward(
            prices_tlt, lambda _, m=ma: SellInMayBondConfirm(tlt_ma_period=m), folds=folds_tlt
        )
        print(f"    {ma:>3}d  Sharpe {w.combined_report.sharpe:.3f}")

    print(section("H044: JanuaryBarometerGoldFilter  canonical (GLD < 50MA)"))
    wf44, rep44, lo44, hi44 = run(prices_gld, folds_gld, JanuaryBarometerGoldFilter, "canonical")
    print("  Sensitivity (gld_ma_period):")
    for ma in [20, 50, 100, 200]:
        w = walk_forward(
            prices_gld,
            lambda _, m=ma: JanuaryBarometerGoldFilter(gld_ma_period=m),
            folds=folds_gld,
        )
        print(f"    {ma:>3}d  Sharpe {w.combined_report.sharpe:.3f}")

    print(section("H045: TurnOfMonthLowVol  canonical (3d window, VIX < 25)"))
    wf45, rep45, lo45, hi45 = run(prices_vix, folds_vix, TurnOfMonthLowVol, "canonical")
    print("  Sensitivity (vix_threshold):")
    for vt in [20, 25, 30, 35]:
        w = walk_forward(
            prices_vix, lambda _, v=vt: TurnOfMonthLowVol(vix_threshold=v), folds=folds_vix
        )
        print(f"    vix<{vt}  Sharpe {w.combined_report.sharpe:.3f}")

    print(section("H046: HalloweenMomentumConfirm  canonical (6m momentum > 0)"))
    wf46, rep46, lo46, hi46 = run(spy, folds_spy, HalloweenMomentumConfirm, "canonical")
    print("  Sensitivity (momentum_days):")
    for md in [63, 126, 189, 252]:
        w = walk_forward(
            spy, lambda _, m=md: HalloweenMomentumConfirm(momentum_days=m), folds=folds_spy
        )
        print(f"    {md:>3}d  Sharpe {w.combined_report.sharpe:.3f}")

    print(section("H047: PreEarningsDrift  canonical (5d, Jan/Apr/Jul/Oct)"))
    wf47, rep47, lo47, hi47 = run(spy, folds_spy, PreEarningsDrift, "canonical")
    print("  Sensitivity (n_days):")
    for nd in [3, 5, 7, 10]:
        w = walk_forward(spy, lambda _, n=nd: PreEarningsDrift(n_days=n), folds=folds_spy)
        print(f"    {nd:>2}d  Sharpe {w.combined_report.sharpe:.3f}")

    print(section("CORRELATION WITH H001 (net returns, same folds)"))
    h001_ret = wf_h001.combined_returns
    strategies = [
        ("H043", wf43, folds_tlt),
        ("H044", wf44, folds_gld),
        ("H045", wf45, folds_vix),
        ("H046", wf46, folds_spy),
        ("H047", wf47, folds_spy),
    ]
    for name, wf, _ in strategies:
        corr = wf.combined_returns.corr(h001_ret.reindex(wf.combined_returns.index))
        print(f"  {name}: {corr:+.4f}")

    print(section("DECISION SUMMARY"))
    results = [
        ("H043", rep43, lo43, hi43),
        ("H044", rep44, lo44, hi44),
        ("H045", rep45, lo45, hi45),
        ("H046", rep46, lo46, hi46),
        ("H047", rep47, lo47, hi47),
    ]
    for name, rep, lo, hi in results:
        decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"
        print(f"  {name}: Sharpe {rep.sharpe:.3f} CI [{lo:.3f}, {hi:.3f}]  -> {decision}")

    print(section("PER-FOLD SHARPE"))
    for name, wf, _ in strategies:
        fold_sharpes = [f"{sharpe(fr.net_returns):+.2f}" for fr in wf.fold_results]
        print(f"  {name}: {' | '.join(fold_sharpes)}")


if __name__ == "__main__":
    main()
