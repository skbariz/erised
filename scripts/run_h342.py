"""H342: VRPRegimeSleevesAllocator -- three-sleeve blend (H086 / H021 / H078) via VRP percentile.

High VRP (>67th pctile): 60% H086 / 30% H021 / 10% H078
Low VRP (<33rd pctile): 10% H086 / 40% H021 / 50% H078
Normal: 30% H086 / 50% H021 / 20% H078

Run: uv run python scripts/run_h342.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from write_summary import write_summary

from quant.data.loader import load_research
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import (
    LowVolTargetPortfolio,
    MovingAverageCrossover,
    RiskParity,
    SeasonalRiskParity,
)
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})
    prices = spy.join(tlt, how="left").join(vix, how="left")
    for col in ["tlt_close", "vix_close"]:
        prices[col] = prices[col].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H342: VRPRegimeSleevesAllocator (H086/H021/H078 via VRP percentile)\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    # --- run component strategies ---
    wf_h086 = walk_forward_multi(prices, lambda _: LowVolTargetPortfolio(), folds=folds)
    wf_h021 = walk_forward_multi(prices, lambda _: RiskParity(), folds=folds)
    wf_h078 = walk_forward_multi(prices, lambda _: SeasonalRiskParity(), folds=folds)
    ret_h086 = wf_h086.combined_returns
    ret_h021 = wf_h021.combined_returns
    ret_h078 = wf_h078.combined_returns

    print(
        f"\nH086 standalone: Sharpe {wf_h086.combined_report.sharpe:.3f}  MaxDD {wf_h086.combined_report.max_drawdown:.2%}"
    )
    print(
        f"H021 standalone: Sharpe {wf_h021.combined_report.sharpe:.3f}  MaxDD {wf_h021.combined_report.max_drawdown:.2%}"
    )
    print(
        f"H078 standalone: Sharpe {wf_h078.combined_report.sharpe:.3f}  MaxDD {wf_h078.combined_report.max_drawdown:.2%}"
    )

    # --- VRP percentile signal (252d rolling, lagged 1 day) ---
    spy_r = spy["adj_close"].pct_change()
    vix_var = (prices["vix_close"] / 100) ** 2
    realized_var = spy_r.rolling(21).var() * 252
    vrp = vix_var - realized_var

    vrp_pctile = vrp.rolling(252).rank(pct=True).shift(1)  # lagged

    high_vrp = (vrp_pctile > 0.67).astype(float)
    low_vrp = (vrp_pctile < 0.33).astype(float)

    # weights for each sleeve
    idx = ret_h086.index
    w_h086 = pd.Series(0.30, index=prices.index)
    w_h086 = w_h086.where(~high_vrp.astype(bool), 0.60)
    w_h086 = w_h086.where(~low_vrp.astype(bool), 0.10)

    w_h021 = pd.Series(0.50, index=prices.index)
    w_h021 = w_h021.where(~high_vrp.astype(bool), 0.30)
    w_h021 = w_h021.where(~low_vrp.astype(bool), 0.40)

    w_h078 = pd.Series(0.20, index=prices.index)
    w_h078 = w_h078.where(~high_vrp.astype(bool), 0.10)
    w_h078 = w_h078.where(~low_vrp.astype(bool), 0.50)

    w1 = w_h086.reindex(idx).fillna(0.30)
    w2 = w_h021.reindex(idx).fillna(0.50)
    w3 = w_h078.reindex(idx).fillna(0.20)

    blended_rets = (
        w1 * ret_h086
        + w2 * ret_h021.reindex(idx).fillna(0.0)
        + w3 * ret_h078.reindex(idx).fillna(0.0)
    )

    lo, hi = bootstrap_sharpe_ci(blended_rets, n_boot=5000)
    bl_sharpe = float(sharpe(blended_rets))
    bl_cagr = float((1 + blended_rets).prod() ** (252 / len(blended_rets)) - 1)
    cum = (1 + blended_rets).cumprod()
    bl_maxdd = float((cum / cum.cummax() - 1).min())
    bl_ntrades = int(
        wf_h086.combined_report.n_trades
        + wf_h021.combined_report.n_trades
        + wf_h078.combined_report.n_trades
    )

    print("\nBlended result:")
    print(f"  Sharpe {bl_sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {bl_cagr:.2%}  MaxDD {bl_maxdd:.2%}  N_trades {bl_ntrades}")

    # --- regime distribution ---
    vrp_pctile_oos = vrp_pctile.reindex(idx)
    n_high = (vrp_pctile_oos > 0.67).sum()
    n_low = (vrp_pctile_oos < 0.33).sum()
    n_normal = len(idx) - n_high - n_low
    print("\nRegime distribution (OOS):")
    print(
        f"  High VRP: {n_high / len(idx):.1%}  Low VRP: {n_low / len(idx):.1%}  Normal: {n_normal / len(idx):.1%}"
    )

    # --- per-fold Sharpe ---
    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds):
        fold_rets = blended_rets.loc[fold.test_start : fold.test_end]
        fs = float(sharpe(fold_rets)) if len(fold_rets) > 0 else 0.0
        fold_sharpes.append(fs)
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    # --- H001 correlation ---
    folds_spy = make_folds(spy, train_years=5, test_years=2)
    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_spy)
    corr = float(blended_rets.corr(wf_h001.combined_returns.reindex(blended_rets.index)))
    print(f"\nH001 correlation: {corr:+.4f}")

    decision = "KEEP" if lo > 0 and bl_sharpe > 0.51 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="342",
        name="vrp-regime-sleeves-allocator",
        strategy_class="VRPRegimeSleevesAllocator",
        family="sleeve allocation",
        sharpe=bl_sharpe,
        ci_low=lo,
        ci_high=hi,
        cagr=bl_cagr,
        max_drawdown=bl_maxdd,
        n_trades=bl_ntrades,
        h001_corr=corr,
        fold_sharpes=fold_sharpes,
        decision=decision,
        notes=(
            "Runner-level ensemble: H086 + H021 + H078 blended via 252d rolling VRP percentile. "
            "High VRP (>67th): 60/30/10; Low VRP (<33rd): 10/40/50; Normal: 30/50/20. "
            "Percentile-based threshold adapts to structural VRP changes over time. "
            "Signal lagged 1 day to avoid lookahead."
        ),
        canonical_params={
            "high_vrp_threshold": 0.67,
            "low_vrp_threshold": 0.33,
            "vrp_pctile_window": 252,
            "vrp_realized_window": 21,
        },
    )


if __name__ == "__main__":
    main()
