"""Batch 6 walk-forward runner: H033-H037 (cross-asset macro timing).

Run: uv run python scripts/run_batch6.py

All five strategies use TLT and/or GLD as signals to time SPY exposure.
Goal: H001 correlation below 0.40 for ensemble diversification.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import (
    BondTrendEquityTimer,
    DualSafeHavenConfirmation,
    EquityVsSafeHavenMomentum,
    GoldFearGauge,
    MovingAverageCrossover,
    SPYTLTCorrelationRegime,
)
from quant.validation.walk_forward import make_folds, walk_forward

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

    prices_tlt = spy.join(tlt, how="left")
    prices_tlt["tlt_close"] = prices_tlt["tlt_close"].ffill()

    prices_gld = spy.join(gld, how="left")
    prices_gld["gld_close"] = prices_gld["gld_close"].ffill()

    prices_all = spy.join(tlt, how="left").join(gld, how="left")
    prices_all["tlt_close"] = prices_all["tlt_close"].ffill()
    prices_all["gld_close"] = prices_all["gld_close"].ffill()

    folds_tlt = make_folds(prices_tlt, train_years=5, test_years=2)
    folds_gld = make_folds(prices_gld, train_years=5, test_years=2)
    folds_all = make_folds(prices_all, train_years=5, test_years=2)

    print(f"Batch 6: H033-H037 — Cross-Asset Macro Timing\n{SEP}")
    print(
        f"SPY+TLT rows: {len(prices_tlt)}  SPY+GLD rows: {len(prices_gld)}  SPY+TLT+GLD rows: {len(prices_all)}"
    )

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

    print(section("H033: BondTrendEquityTimer  canonical (ma_period=200d)"))
    wf33, rep33, lo33, hi33 = run(prices_tlt, folds_tlt, BondTrendEquityTimer, "canonical")
    print("  Sensitivity (ma_period):")
    for mp in [50, 100, 150, 200, 250]:
        w = walk_forward(
            prices_tlt, lambda _, m=mp: BondTrendEquityTimer(ma_period=m), folds=folds_tlt
        )
        print(f"    ma_period={mp:>4}d  Sharpe {w.combined_report.sharpe:.3f}")

    print(section("H034: SPYTLTCorrelationRegime  canonical (corr_window=60d)"))
    wf34, rep34, lo34, hi34 = run(prices_tlt, folds_tlt, SPYTLTCorrelationRegime, "canonical")
    print("  Sensitivity (corr_window):")
    for cw in [20, 40, 60, 120, 252]:
        w = walk_forward(
            prices_tlt, lambda _, c=cw: SPYTLTCorrelationRegime(corr_window=c), folds=folds_tlt
        )
        print(f"    corr_window={cw:>4}d  Sharpe {w.combined_report.sharpe:.3f}")

    print(section("H035: GoldFearGauge  canonical (ma_period=50d)"))
    wf35, rep35, lo35, hi35 = run(prices_gld, folds_gld, GoldFearGauge, "canonical")
    print("  Sensitivity (ma_period):")
    for mp in [20, 50, 100, 200]:
        w = walk_forward(prices_gld, lambda _, m=mp: GoldFearGauge(ma_period=m), folds=folds_gld)
        print(f"    ma_period={mp:>4}d  Sharpe {w.combined_report.sharpe:.3f}")

    print(section("H036: EquityVsSafeHavenMomentum  canonical (lookback_days=63d)"))
    wf36, rep36, lo36, hi36 = run(prices_all, folds_all, EquityVsSafeHavenMomentum, "canonical")
    print("  Sensitivity (lookback_days):")
    for lb in [21, 42, 63, 126, 252]:
        w = walk_forward(
            prices_all,
            lambda _, lb_=lb: EquityVsSafeHavenMomentum(lookback_days=lb_),
            folds=folds_all,
        )
        print(f"    lookback_days={lb:>4}d  Sharpe {w.combined_report.sharpe:.3f}")

    print(section("H037: DualSafeHavenConfirmation  canonical (ma_period=200d)"))
    wf37, rep37, lo37, hi37 = run(prices_all, folds_all, DualSafeHavenConfirmation, "canonical")
    print("  Sensitivity (ma_period):")
    for mp in [50, 100, 150, 200, 250]:
        w = walk_forward(
            prices_all,
            lambda _, m=mp: DualSafeHavenConfirmation(ma_period=m),
            folds=folds_all,
        )
        print(f"    ma_period={mp:>4}d  Sharpe {w.combined_report.sharpe:.3f}")

    print(section("CORRELATION WITH H001 (net returns, same folds)"))
    h001_ret = wf_h001.combined_returns
    strategies = [
        ("H033", wf33, folds_tlt),
        ("H034", wf34, folds_tlt),
        ("H035", wf35, folds_gld),
        ("H036", wf36, folds_all),
        ("H037", wf37, folds_all),
    ]
    for name, wf, _ in strategies:
        corr = wf.combined_returns.corr(h001_ret.reindex(wf.combined_returns.index))
        print(f"  {name}: {corr:+.4f}")

    print(section("DECISION SUMMARY"))
    results = [
        ("H033", rep33, lo33, hi33),
        ("H034", rep34, lo34, hi34),
        ("H035", rep35, lo35, hi35),
        ("H036", rep36, lo36, hi36),
        ("H037", rep37, lo37, hi37),
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
