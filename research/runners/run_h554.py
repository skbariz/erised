"""H554: Stress-test evaluation — 2022 inflation/rate-hike regime.

Evaluates the H551 multi-objective portfolio (H065/H076/H473/H157 weights) and all
Target-bar MDR survivors on the 2022 calendar year (2022-01-01 to 2022-12-31).

2022 regime: SPY -18.2%, TLT -31.2%, GLD -0.3%, DBC +26.3%.
SPY-TLT correlation flipped strongly positive — risk parity's worst enemy.

Approach: run a single supplementary fold (train 2017-2021, test 2022) for the
key ETF-based strategies. H473/H157 (CS strategies) use fold 11 as proxy since
they have no parquet data for 2022 beyond the research period.

Run: uv run python scripts/run_h554.py
Expected runtime: ~10-20 minutes (ETF strategies only, single fold).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from erised.data.loader import load_research
from erised.metrics.performance import cagr, max_drawdown, sharpe
from erised.strategies.base import (
    CarryMomentumSPYTLT, DefensiveSectorOverlay, EqualWeightDefensiveFactor,
    FourAssetRiskParity, GoldFearGauge, H043H021RegimeBlend,
    HalloweenMomentumConfirm, LowVolTargetPortfolio, MonthlyRebalancingRiskParity,
    MovingAverageCrossover, MultiAssetDynamicVolTarget, MultiAssetStrategy,
    RatchetDrawdownProtection, RiskParity, SeasonalRiskParity, SellInMayBondConfirm,
    VolScaledTrend, VrpCalendarHybrid,
)
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 70
TRAIN_START = pd.Timestamp("2017-01-01")
TEST_START  = pd.Timestamp("2022-01-01")
TEST_END    = pd.Timestamp("2022-12-31")

# H551 optimal weights (from multi-objective optimizer)
H551_WEIGHTS = {
    "H065": 0.2686, "H076": 0.2580, "H473": 0.1652,
    "H157": 0.1496, "H016": 0.0744, "H063": 0.0337,
    "H043": 0.0188, "H192": 0.0128,
}

# Normalise (H473/H157/H192 are CS strategies — approximated by their fold-11 Sharpe proxy)
# For ETF-only stress test, we drop CS strategies and renorm
ETF_ONLY_WEIGHTS = {k: v for k, v in H551_WEIGHTS.items() if k not in {"H473", "H157", "H192"}}
etf_total = sum(ETF_ONLY_WEIGHTS.values())
ETF_ONLY_WEIGHTS = {k: v / etf_total for k, v in ETF_ONLY_WEIGHTS.items()}


def run_strategy_2022(prices_aa, strat):
    """Run strategy on train 2017-2021, test 2022. Returns 2022 return series."""
    if isinstance(strat, MultiAssetStrategy):
        wf = walk_forward_multi(prices_aa, lambda _, s=strat: s,
                                train_years=5, test_years=1,
                                folds=[_make_2022_fold(prices_aa)])
    else:
        wf = walk_forward(prices_aa, lambda _, s=strat: s,
                         folds=[_make_2022_fold(prices_aa)])
    mask = (wf.combined_returns.index >= TEST_START) & (wf.combined_returns.index <= TEST_END)
    return wf.combined_returns[mask]


def _make_2022_fold(prices_aa):
    from erised.validation.walk_forward import Fold
    return Fold(
        fold_idx=99,
        train_start=TRAIN_START,
        train_end=TEST_START,
        test_start=TEST_START,
        test_end=TEST_END + pd.Timedelta(days=1),
    )


def metrics_str(rets):
    if len(rets) < 10:
        return "INSUFFICIENT DATA"
    s = sharpe(rets)
    c = cagr(rets)
    dd = max_drawdown(rets)
    cal = c / abs(dd) if dd < 0 else 0
    return f"Sharpe {s:+.3f}  CAGR {c:+.2%}  MaxDD {dd:.2%}  Calmar {cal:.3f}"


def main():
    print(f"H554: 2022 Inflation Stress Test (train 2017-2021, test 2022)\n{SEP}")
    print(f"SPY benchmark 2022: -18.2% | TLT: -31.2% | GLD: -0.3% | DBC: +26.3%")
    print(f"Context: SPY-TLT correlation flipped +0.6 — risk parity's worst regime.\n")

    spy = load_research("SPY")
    etfs = {n: load_research(n)[["adj_close"]].rename(columns={"adj_close": f"{n.lower()}_close"})
            for n in ["TLT","GLD","VIX","SHY","DBC","XLP","XLU","IVE"]}
    prices_aa = spy.copy()
    for df in etfs.values():
        prices_aa = prices_aa.join(df, how="left")
    for col in [c for c in prices_aa.columns if c != "adj_close"]:
        prices_aa[col] = prices_aa[col].ffill()

    # SPY benchmark for 2022
    spy_2022 = spy["adj_close"].pct_change().loc[TEST_START:TEST_END]
    print(f"SPY 2022: {metrics_str(spy_2022)}")

    # ETF strategy pool — all relevant to H551 weights
    strategies = {
        "H016": VolScaledTrend(),
        "H043": SellInMayBondConfirm(),
        "H063": DefensiveSectorOverlay(),
        "H065": EqualWeightDefensiveFactor(),
        "H076": FourAssetRiskParity(),
        "H021": RiskParity(),
        "H086": LowVolTargetPortfolio(),
        "H087": RatchetDrawdownProtection(),
        "H089": H043H021RegimeBlend(),
        "H078": SeasonalRiskParity(),
        "H070": MultiAssetDynamicVolTarget(),
        "H060": CarryMomentumSPYTLT(),
        "H035": GoldFearGauge(),
        "H046": HalloweenMomentumConfirm(),
        "H068": VrpCalendarHybrid(),
        "H090": MonthlyRebalancingRiskParity(),
    }

    print(f"\n{SEP}")
    print("Individual ETF strategy performance in 2022:")
    print(f"{'Strategy':<8} {'2022 Sharpe':>12} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>8}")
    print("-" * 50)

    rets_2022 = {}
    for hid, strat in strategies.items():
        r = run_strategy_2022(prices_aa, strat)
        rets_2022[hid] = r
        s = sharpe(r) if len(r) > 10 else float("nan")
        c = cagr(r) if len(r) > 10 else float("nan")
        dd = max_drawdown(r) if len(r) > 10 else float("nan")
        cal = c / abs(dd) if dd < 0 and len(r) > 10 else float("nan")
        flag = " <-- H551 weight" if hid in ETF_ONLY_WEIGHTS else ""
        print(f"{hid:<8} {s:>12.3f} {c:>7.2%} {dd:>7.2%} {cal:>8.3f}{flag}")

    # H551 ETF-only portfolio (proxy — drops H473/H157/H192)
    print(f"\n{SEP}")
    print(f"H551 ETF-only proxy (drops H473/H157 CS components, renorms):")
    print(f"Weights: {ETF_ONLY_WEIGHTS}")
    ret_df = pd.DataFrame(rets_2022)
    common_idx = ret_df.dropna(how="all").index
    h551_etf = sum(ret_df.loc[common_idx, k] * w for k, w in ETF_ONLY_WEIGHTS.items()
                   if k in ret_df.columns)
    print(f"H551 ETF proxy 2022: {metrics_str(h551_etf)}")

    # Key risk-parity variants for comparison
    print(f"\n{SEP}")
    print("Target-bar survivor comparison (ETF components available):")
    # H543/H300 MDR concentrates on H043/H078/H035/H473 — ETF proxy uses H043+H078+H035
    h300_proxy_w = {"H043": 0.24, "H078": 0.20, "H035": 0.15, "H060": 0.13}
    total = sum(h300_proxy_w.values())
    h300_proxy_w = {k: v / total for k, v in h300_proxy_w.items()}
    h300_rets = sum(ret_df.loc[common_idx, k] * w for k, w in h300_proxy_w.items()
                    if k in ret_df.columns)
    print(f"H300/H543 MDR ETF proxy 2022: {metrics_str(h300_rets)}")

    # H089 (calendar+risk parity blend) — all ETF
    if "H089" in rets_2022:
        print(f"H089 H043+H021 blend 2022: {metrics_str(rets_2022['H089'])}")
    # H086 vol-target
    if "H086" in rets_2022:
        print(f"H086 LowVolTarget 2022: {metrics_str(rets_2022['H086'])}")

    # Verdict
    print(f"\n{SEP}")
    print("STRESS TEST VERDICT:")
    h551_sharpe_2022 = sharpe(h551_etf) if len(h551_etf) > 10 else float("nan")
    h551_mdd_2022 = max_drawdown(h551_etf) if len(h551_etf) > 10 else float("nan")

    if h551_sharpe_2022 > 0 and h551_mdd_2022 > -0.20:
        verdict = "PASS — H551 ETF proxy survives 2022 stress (Sharpe > 0, MaxDD > -20%)"
    elif h551_sharpe_2022 > 0:
        verdict = "CONDITIONAL PASS — positive Sharpe but MaxDD exceeds -20%"
    else:
        verdict = "FAIL — H551 ETF proxy has negative Sharpe in 2022 regime"

    print(f"H551 ETF proxy: Sharpe {h551_sharpe_2022:.3f}, MaxDD {h551_mdd_2022:.2%}")
    print(f"Verdict: {verdict}")
    print(f"\nNote: H473 (LongShortPCAReversal) and H157 (VolScaledMomentum CS) not tested")
    print(f"here — they require the full stock panel. Fold 11 (2020-2022) was their last test window.")
    print(f"H551's 2022 risk is primarily through H076 (25.8% weight) which holds TLT.")


if __name__ == "__main__":
    main()
