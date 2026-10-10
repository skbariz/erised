"""Batch 1 walk-forward runner: H003-H012.

Run: uv run python scripts/run_batch1.py

Evaluates all 10 strategies with the same folds as H001/H002 and writes
results to stdout. Copy numbers into research logs after review.
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies.base import (
    AbsoluteMomentumStrategy,
    ADXFilteredMACrossover,
    ConsecutiveDownDays,
    FastMACrossover,
    MACDCrossover,
    MovingAverageCrossover,
    RSIReversionStrategy,
    ShortHoldMeanReversion,
    TimeSeriesMomentum,
    VIXFilteredReversionStrategy,
    VolatilityRegime,
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

    # VIX for H006
    vix_path = Path(__file__).resolve().parents[2] / "data" / "VIX_research.parquet"
    vix = pd.read_parquet(vix_path)[["adj_close"]].rename(columns={"adj_close": "vix_close"})
    prices_vix = prices.join(vix, how="left")
    prices_vix["vix_close"] = prices_vix["vix_close"].ffill()

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

    # ------------------------------------------------------------------ H005
    print(section("H005: ConsecutiveDownDays  canonical (n_down=3, hold_days=3)"))
    wf5, rep5, lo5, hi5 = run(prices, folds, ConsecutiveDownDays, "canonical")
    print("  Grid (n_down x hold_days):")
    for nd, hd in product([2, 3, 4], [2, 3, 5]):
        wf = walk_forward(prices, lambda _, n=nd, h=hd: ConsecutiveDownDays(n, h), folds=folds)
        print(f"    n_down={nd} hold={hd}  Sharpe {wf.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ H006
    print(
        section(
            "H006: VIXFilteredReversionStrategy  canonical (lookback=20, thresh=2.0, hold=2, vix=20)"
        )
    )
    wf6, rep6, lo6, hi6 = run(prices_vix, folds, VIXFilteredReversionStrategy, "canonical")
    print("  1-D sensitivity (vix_threshold):")
    for vt in [15, 20, 25]:
        wf = walk_forward(
            prices_vix, lambda _, v=vt: VIXFilteredReversionStrategy(vix_threshold=v), folds=folds
        )
        print(f"    vix_threshold={vt}  Sharpe {wf.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ H007
    print(section("H007: TimeSeriesMomentum  canonical (12m-1m skip)"))
    wf7, rep7, lo7, hi7 = run(prices, folds, TimeSeriesMomentum, "canonical")
    print("  1-D sensitivity (long_months):")
    for lm in [6, 9, 12]:
        wf = walk_forward(prices, lambda _, l=lm: TimeSeriesMomentum(long_months=l), folds=folds)
        print(f"    long_months={lm}  Sharpe {wf.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ H008
    print(section("H008: AbsoluteMomentumStrategy  canonical (12m lookback)"))
    wf8, rep8, lo8, hi8 = run(prices, folds, AbsoluteMomentumStrategy, "canonical")
    print("  1-D sensitivity (lookback_months):")
    for lb in [6, 9, 12]:
        wf = walk_forward(prices, lambda _, l=lb: AbsoluteMomentumStrategy(l), folds=folds)
        print(f"    lookback_months={lb}  Sharpe {wf.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ H009
    print(section("H009: MACDCrossover  canonical (12,26,9)"))
    wf9, rep9, lo9, hi9 = run(prices, folds, MACDCrossover, "canonical")
    print("  Grid variants:")
    for f, s, sig in [(12, 26, 9), (8, 21, 5), (5, 13, 3)]:
        wf = walk_forward(
            prices, lambda _, ff=f, ss=s, sg=sig: MACDCrossover(ff, ss, sg), folds=folds
        )
        print(f"    MACD({f},{s},{sig})  Sharpe {wf.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ H010
    print(section("H010: FastMACrossover  canonical (10,50)"))
    wf10, rep10, lo10, hi10 = run(prices, folds, FastMACrossover, "canonical")
    print("  Grid variants:")
    for f, s in [(10, 50), (20, 100), (5, 20)]:
        wf = walk_forward(prices, lambda _, ff=f, ss=s: FastMACrossover(ff, ss), folds=folds)
        print(f"    MA({f},{s})  Sharpe {wf.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ H011
    print(section("H011: ADXFilteredMACrossover  canonical (MA50/200, ADX14>25)"))
    wf11, rep11, lo11, hi11 = run(prices, folds, ADXFilteredMACrossover, "canonical")
    print("  1-D sensitivity (adx_threshold):")
    for thr in [20, 25, 30]:
        wf = walk_forward(
            prices, lambda _, t=thr: ADXFilteredMACrossover(adx_threshold=t), folds=folds
        )
        print(f"    adx_threshold={thr}  Sharpe {wf.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ H012
    print(section("H012: VolatilityRegime  canonical (vol_lookback=20, regime=252)"))
    wf12, rep12, lo12, hi12 = run(prices, folds, VolatilityRegime, "canonical")
    print("  Grid (vol_lookback x regime_lookback):")
    for vl, rl in product([10, 20, 30], [126, 252]):
        wf = walk_forward(prices, lambda _, v=vl, r=rl: VolatilityRegime(v, r), folds=folds)
        print(f"    vol={vl} regime={rl}  Sharpe {wf.combined_report.sharpe:.2f}")

    # ------------------------------------------------------------------ Correlation table
    print(section("CORRELATION WITH H001 (net returns, same folds)"))
    strategies = [
        ("H003", wf3),
        ("H004", wf4),
        ("H005", wf5),
        ("H006", wf6),
        ("H007", wf7),
        ("H008", wf8),
        ("H009", wf9),
        ("H010", wf10),
        ("H011", wf11),
        ("H012", wf12),
    ]
    h001_ret = wf_h001.combined_returns
    print(f"  {'Strategy':<8} {'Sharpe':>8} {'vs H001':>10}")
    for name, wf in strategies:
        corr = wf.combined_returns.corr(h001_ret)
        print(f"  {name:<8} {wf.combined_report.sharpe:>8.2f} {corr:>+10.3f}")


if __name__ == "__main__":
    main()
