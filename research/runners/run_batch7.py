"""Batch 7 walk-forward runner: H038-H042 (contrarian & mean reversion).

Run: uv run python scripts/run_batch7.py

All five strategies fire when H001 is dormant or wrong — structurally
opposite to trend-following. Target H001 correlation: -0.30 to +0.15.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies import (
    DeepDrawdownCrisisBuyer,
    LongTermReversal,
    MovingAverageCrossover,
    PostCrashRecovery,
    ShortTermWeeklyReversal,
    VolOfVolCalm,
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
    prices_vix = spy.join(vix, how="left")
    prices_vix["vix_close"] = prices_vix["vix_close"].ffill()

    folds_spy = make_folds(spy, train_years=5, test_years=2)
    folds_vix = make_folds(prices_vix, train_years=5, test_years=2)

    print(f"Batch 7: H038-H042 — Contrarian & Mean Reversion\n{SEP}")
    print(f"SPY rows: {len(spy)}  SPY+VIX rows: {len(prices_vix)}")

    print(section("H001 baseline (reference)"))
    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_spy)
    lo_h001, hi_h001 = bootstrap_sharpe_ci(wf_h001.combined_returns, n_boot=5000)
    print(
        f"  [H001]  Sharpe {wf_h001.combined_report.sharpe:.3f}  CI [{lo_h001:.3f}, {hi_h001:.3f}]"
        f"  CAGR {wf_h001.combined_report.cagr:.2%}  MaxDD {wf_h001.combined_report.max_drawdown:.2%}"
    )

    print(section("H038: LongTermReversal  canonical (lookback=1260d, percentile=33)"))
    wf38, rep38, lo38, hi38 = run(spy, folds_spy, LongTermReversal, "canonical")
    print("  Sensitivity (lookback_days):")
    for lb in [504, 756, 1008, 1260, 1512]:
        w = walk_forward(spy, lambda _, lb_=lb: LongTermReversal(lookback_days=lb_), folds=folds_spy)
        print(f"    {lb:>5}d  Sharpe {w.combined_report.sharpe:.3f}")

    print(section("H039: DeepDrawdownCrisisBuyer  canonical (entry=-20%, exit=-5%)"))
    wf39, rep39, lo39, hi39 = run(spy, folds_spy, DeepDrawdownCrisisBuyer, "canonical")
    print("  Sensitivity (entry_dd):")
    for ed in [-0.10, -0.15, -0.20, -0.25, -0.30]:
        w = walk_forward(spy, lambda _, e=ed: DeepDrawdownCrisisBuyer(entry_dd=e), folds=folds_spy)
        print(f"    entry={ed:.0%}  Sharpe {w.combined_report.sharpe:.3f}")

    print(section("H040: ShortTermWeeklyReversal  canonical (hold=5d, threshold=10th pct)"))
    wf40, rep40, lo40, hi40 = run(spy, folds_spy, ShortTermWeeklyReversal, "canonical")
    print("  Sensitivity (threshold):")
    for thr in [5, 10, 15, 20]:
        w = walk_forward(
            spy, lambda _, t=thr: ShortTermWeeklyReversal(threshold=t), folds=folds_spy
        )
        print(f"    p{thr:<3}  Sharpe {w.combined_report.sharpe:.3f}")

    print(section("H041: PostCrashRecovery  canonical (vix=40, delay=30d, hold=60d)"))
    wf41, rep41, lo41, hi41 = run(prices_vix, folds_vix, PostCrashRecovery, "canonical")
    print("  Sensitivity (vix_threshold):")
    for vt in [30, 35, 40, 45]:
        w = walk_forward(
            prices_vix, lambda _, v=vt: PostCrashRecovery(vix_threshold=v), folds=folds_vix
        )
        print(f"    vix={vt}  Sharpe {w.combined_report.sharpe:.3f}")

    print(section("H042: VolOfVolCalm  canonical (vov_window=20d, percentile=33)"))
    wf42, rep42, lo42, hi42 = run(prices_vix, folds_vix, VolOfVolCalm, "canonical")
    print("  Sensitivity (vov_window):")
    for vw in [10, 20, 40, 60]:
        w = walk_forward(prices_vix, lambda _, v=vw: VolOfVolCalm(vov_window=v), folds=folds_vix)
        print(f"    {vw:>3}d  Sharpe {w.combined_report.sharpe:.3f}")

    print(section("CORRELATION WITH H001 (net returns, same folds)"))
    h001_ret = wf_h001.combined_returns
    strategies = [
        ("H038", wf38, folds_spy),
        ("H039", wf39, folds_spy),
        ("H040", wf40, folds_spy),
        ("H041", wf41, folds_vix),
        ("H042", wf42, folds_vix),
    ]
    for name, wf, _ in strategies:
        corr = wf.combined_returns.corr(h001_ret.reindex(wf.combined_returns.index))
        print(f"  {name}: {corr:+.4f}")

    print(section("DECISION SUMMARY"))
    results = [
        ("H038", rep38, lo38, hi38),
        ("H039", rep39, lo39, hi39),
        ("H040", rep40, lo40, hi40),
        ("H041", rep41, lo41, hi41),
        ("H042", rep42, lo42, hi42),
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
