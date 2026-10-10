"""H341: TwoRegimeSleeveAllocator -- two-sleeve blend (H021 / H078) via VRP + term-spread regime.

Favorable (VRP>0 AND TLT 252d > SHY 252d): 70% H021 / 30% H078
Unfavorable (VRP<0 AND TLT 252d < SHY 252d): 30% H021 / 70% H078
Mixed: 50% H021 / 50% H078

Run: uv run python scripts/run_h341.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from write_summary import write_summary

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies.base import (
    MovingAverageCrossover,
    RiskParity,
    SeasonalRiskParity,
)
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    shy = load_research("SHY")[["adj_close"]].rename(columns={"adj_close": "shy_close"})
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})
    prices = spy.join(tlt, how="left").join(shy, how="left").join(vix, how="left")
    for col in ["tlt_close", "shy_close", "vix_close"]:
        prices[col] = prices[col].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H341: TwoRegimeSleeveAllocator (H021/H078 blend via VRP + term spread)\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    # --- run component strategies ---
    wf_h021 = walk_forward_multi(prices, lambda _: RiskParity(), folds=folds)
    wf_h078 = walk_forward_multi(prices, lambda _: SeasonalRiskParity(), folds=folds)
    ret_h021 = wf_h021.combined_returns
    ret_h078 = wf_h078.combined_returns

    print(
        f"\nH021 standalone: Sharpe {wf_h021.combined_report.sharpe:.3f}  MaxDD {wf_h021.combined_report.max_drawdown:.2%}"
    )
    print(
        f"H078 standalone: Sharpe {wf_h078.combined_report.sharpe:.3f}  MaxDD {wf_h078.combined_report.max_drawdown:.2%}"
    )

    # --- regime signal on full price history (lagged 1 day to avoid lookahead) ---
    spy_r = spy["adj_close"].pct_change()
    vix_var = (prices["vix_close"] / 100) ** 2
    realized_var = spy_r.rolling(21).var() * 252
    vrp = (vix_var - realized_var).shift(1)  # lagged

    tlt_mom = prices["tlt_close"].pct_change(252).shift(1)
    shy_mom = prices["shy_close"].pct_change(252).shift(1)
    term_spread_positive = (tlt_mom > shy_mom).astype(float)

    vrp_positive = (vrp > 0).astype(float)

    # regime classification
    favorable = vrp_positive * term_spread_positive  # both positive
    unfavorable = ((vrp <= 0) & (tlt_mom <= shy_mom)).astype(float)

    w_h021 = pd.Series(0.5, index=prices.index)
    w_h021 = w_h021.where(~favorable.astype(bool), 0.70)
    w_h021 = w_h021.where(~unfavorable.astype(bool), 0.30)

    # align to OOS period
    idx = ret_h021.index
    w1 = w_h021.reindex(idx).fillna(0.5)
    w2 = 1.0 - w1

    blended_rets = w1 * ret_h021 + w2 * ret_h078.reindex(idx).fillna(0.0)

    lo, hi = bootstrap_sharpe_ci(blended_rets, n_boot=5000)
    bl_sharpe = float(sharpe(blended_rets))
    bl_cagr = float((1 + blended_rets).prod() ** (252 / len(blended_rets)) - 1)
    cum = (1 + blended_rets).cumprod()
    bl_maxdd = float((cum / cum.cummax() - 1).min())
    bl_ntrades = int(wf_h021.combined_report.n_trades + wf_h078.combined_report.n_trades)

    print("\nBlended result:")
    print(f"  Sharpe {bl_sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {bl_cagr:.2%}  MaxDD {bl_maxdd:.2%}  N_trades {bl_ntrades}")

    # --- regime distribution ---
    oos_favorable = favorable.reindex(idx).sum()
    oos_unfavorable = unfavorable.reindex(idx).sum()
    oos_mixed = len(idx) - oos_favorable - oos_unfavorable
    print("\nRegime distribution (OOS):")
    print(
        f"  Favorable: {oos_favorable / len(idx):.1%}  Unfavorable: {oos_unfavorable / len(idx):.1%}  Mixed: {oos_mixed / len(idx):.1%}"
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
        hypothesis_id="341",
        name="two-regime-sleeve-allocator",
        strategy_class="TwoRegimeSleeveAllocator",
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
            "Runner-level ensemble: H021 + H078 blended via VRP + term-spread regime. "
            "Favorable (VRP>0 AND TLT_252d>SHY_252d): 70/30; "
            "Unfavorable (both negative): 30/70; Mixed: 50/50. "
            "Signal lagged 1 day to avoid lookahead."
        ),
        canonical_params={
            "w_h021_favorable": 0.70,
            "w_h021_unfavorable": 0.30,
            "w_h021_mixed": 0.50,
            "vrp_lookback": 21,
            "mom_lookback": 252,
        },
    )


if __name__ == "__main__":
    main()
