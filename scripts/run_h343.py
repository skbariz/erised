"""H343: StressRegimeDefensiveAllocator -- three-sleeve blend (H021 / H043 / H078) via VIX level.

High stress (VIX > 28): 80% H078 + 20% H043
Medium stress (VIX 20-28): 40% H021 + 30% H043 + 30% H078
Normal (VIX < 20): 70% H021 + 30% H078

Run: uv run python scripts/run_h343.py
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
    MovingAverageCrossover,
    RiskParity,
    SeasonalRiskParity,
    SellInMayBondConfirm,
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
    print(f"H343: StressRegimeDefensiveAllocator (H021/H043/H078 via VIX level)\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    # --- run component strategies ---
    # SellInMayBondConfirm uses only SPY + TLT (single-asset Strategy, not MultiAsset)
    # Use walk_forward_multi for H021 and H078; walk_forward for H043
    wf_h021 = walk_forward_multi(prices, lambda _: RiskParity(), folds=folds)
    wf_h078 = walk_forward_multi(prices, lambda _: SeasonalRiskParity(), folds=folds)

    # H043 needs tlt_close -- pass full prices dataframe
    wf_h043 = walk_forward(prices, lambda _: SellInMayBondConfirm(), folds=folds)

    ret_h021 = wf_h021.combined_returns
    ret_h078 = wf_h078.combined_returns
    ret_h043 = wf_h043.combined_returns

    print(
        f"\nH021 standalone: Sharpe {wf_h021.combined_report.sharpe:.3f}  MaxDD {wf_h021.combined_report.max_drawdown:.2%}"
    )
    print(
        f"H043 standalone: Sharpe {wf_h043.combined_report.sharpe:.3f}  MaxDD {wf_h043.combined_report.max_drawdown:.2%}"
    )
    print(
        f"H078 standalone: Sharpe {wf_h078.combined_report.sharpe:.3f}  MaxDD {wf_h078.combined_report.max_drawdown:.2%}"
    )

    # --- VIX-based regime signal (lagged 1 day to avoid lookahead) ---
    vix_lagged = prices["vix_close"].shift(1)
    high_stress = (vix_lagged > 28).astype(float)
    medium_stress = ((vix_lagged >= 20) & (vix_lagged <= 28)).astype(float)

    # weights for each sleeve
    idx = ret_h021.index

    w_h021 = pd.Series(0.70, index=prices.index)  # normal default
    w_h021 = w_h021.where(~medium_stress.astype(bool), 0.40)
    w_h021 = w_h021.where(~high_stress.astype(bool), 0.00)

    w_h043 = pd.Series(0.00, index=prices.index)  # normal default
    w_h043 = w_h043.where(~medium_stress.astype(bool), 0.30)
    w_h043 = w_h043.where(~high_stress.astype(bool), 0.20)

    w_h078 = pd.Series(0.30, index=prices.index)  # normal default
    w_h078 = w_h078.where(~medium_stress.astype(bool), 0.30)
    w_h078 = w_h078.where(~high_stress.astype(bool), 0.80)

    w1 = w_h021.reindex(idx).fillna(0.70)
    w2 = w_h043.reindex(idx).fillna(0.00)
    w3 = w_h078.reindex(idx).fillna(0.30)

    blended_rets = (
        w1 * ret_h021
        + w2 * ret_h043.reindex(idx).fillna(0.0)
        + w3 * ret_h078.reindex(idx).fillna(0.0)
    )

    lo, hi = bootstrap_sharpe_ci(blended_rets, n_boot=5000)
    bl_sharpe = float(sharpe(blended_rets))
    bl_cagr = float((1 + blended_rets).prod() ** (252 / len(blended_rets)) - 1)
    cum = (1 + blended_rets).cumprod()
    bl_maxdd = float((cum / cum.cummax() - 1).min())
    bl_ntrades = int(
        wf_h021.combined_report.n_trades
        + wf_h043.combined_report.n_trades
        + wf_h078.combined_report.n_trades
    )

    print("\nBlended result:")
    print(f"  Sharpe {bl_sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {bl_cagr:.2%}  MaxDD {bl_maxdd:.2%}  N_trades {bl_ntrades}")

    # --- regime distribution ---
    vix_oos = vix_lagged.reindex(idx)
    n_high = (vix_oos > 28).sum()
    n_medium = ((vix_oos >= 20) & (vix_oos <= 28)).sum()
    n_normal = len(idx) - n_high - n_medium
    print("\nVIX regime distribution (OOS):")
    print(
        f"  High stress (>28): {n_high / len(idx):.1%}  Medium (20-28): {n_medium / len(idx):.1%}  Normal (<20): {n_normal / len(idx):.1%}"
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
        hypothesis_id="343",
        name="stress-regime-defensive-allocator",
        strategy_class="StressRegimeDefensiveAllocator",
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
            "Runner-level ensemble: H021 + H043 + H078 blended via VIX level. "
            "High stress (VIX>28): 0/20/80 H021/H043/H078; "
            "Medium stress (VIX 20-28): 40/30/30; Normal (VIX<20): 70/0/30. "
            "VIX signal lagged 1 day to avoid lookahead. "
            "H043 uses walk_forward (single-asset Strategy); H021/H078 use walk_forward_multi."
        ),
        canonical_params={
            "vix_crisis_threshold": 28,
            "vix_medium_threshold": 20,
            "w_crisis_h021": 0.00,
            "w_crisis_h043": 0.20,
            "w_crisis_h078": 0.80,
        },
    )


if __name__ == "__main__":
    main()
