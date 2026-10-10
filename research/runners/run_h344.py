"""H344: VolTargetedMultiSleeve -- four-sleeve inverse-vol weighted blend, quarterly rebalanced.

Sleeves: H021 (RiskParity), H086 (LowVolTargetPortfolio), H078 (SeasonalRiskParity),
         H089 (H043H021RegimeBlend).
Weights = 1/sigma for each sleeve (252d rolling vol), normalized to sum=1.
Rebalance: first trading day of Jan, Apr, Jul, Oct only.

Run: uv run python scripts/run_h344.py
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
from erised.strategies import (
    H043H021RegimeBlend,
    LowVolTargetPortfolio,
    MovingAverageCrossover,
    RiskParity,
    SeasonalRiskParity,
)
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64

VOL_WINDOW = 252  # trailing vol for inverse-vol weighting
VOL_FLOOR = 1e-6  # avoid divide-by-zero


def quarterly_rebalance_weights(
    rets_dict: dict[str, pd.Series],
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute inverse-vol weights, updated only on the first trading day of each quarter."""
    keys = list(rets_dict.keys())
    # combined DataFrame of returns
    ret_df = pd.DataFrame(rets_dict).reindex(index).fillna(0.0)

    # rolling 252d vol for each sleeve (use full price history up to each date)
    vol_df = ret_df.rolling(VOL_WINDOW).std()

    # identify quarterly rebalance dates: first trading day of Jan/Apr/Jul/Oct
    rebalance_months = {1, 4, 7, 10}
    dates = pd.Series(index)
    is_quarter_start = (dates.dt.month.isin(rebalance_months)) & (
        dates.dt.month != dates.shift(1).dt.month
    )
    rebalance_dates = set(index[is_quarter_start.values])

    # build weight DataFrame: carry forward weights between rebalance dates
    weight_df = pd.DataFrame(index=index, columns=keys, dtype=float)
    current_weights = pd.Series({k: 1.0 / len(keys) for k in keys})  # equal default

    for dt in index:
        if dt in rebalance_dates:
            vols = vol_df.loc[dt]
            if vols.isna().any() or (vols < VOL_FLOOR).any():
                # not enough history yet: equal weight
                current_weights = pd.Series({k: 1.0 / len(keys) for k in keys})
            else:
                inv_vol = 1.0 / vols.clip(lower=VOL_FLOOR)
                current_weights = inv_vol / inv_vol.sum()
        weight_df.loc[dt] = current_weights

    return weight_df.astype(float)


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    prices = spy.join(tlt, how="left")
    prices["tlt_close"] = prices["tlt_close"].ffill()

    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"H344: VolTargetedMultiSleeve (4-sleeve inverse-vol, quarterly rebalance)\n{SEP}")
    print(f"Data: {len(prices)} rows  {prices.index.min().date()} -- {prices.index.max().date()}")
    print(f"Folds: {len(folds)}")

    # --- run component strategies ---
    wf_h021 = walk_forward_multi(prices, lambda _: RiskParity(), folds=folds)
    wf_h086 = walk_forward_multi(prices, lambda _: LowVolTargetPortfolio(), folds=folds)
    wf_h078 = walk_forward_multi(prices, lambda _: SeasonalRiskParity(), folds=folds)
    wf_h089 = walk_forward_multi(prices, lambda _: H043H021RegimeBlend(), folds=folds)

    ret_h021 = wf_h021.combined_returns
    ret_h086 = wf_h086.combined_returns
    ret_h078 = wf_h078.combined_returns
    ret_h089 = wf_h089.combined_returns

    print(
        f"\nH021 standalone: Sharpe {wf_h021.combined_report.sharpe:.3f}  MaxDD {wf_h021.combined_report.max_drawdown:.2%}"
    )
    print(
        f"H086 standalone: Sharpe {wf_h086.combined_report.sharpe:.3f}  MaxDD {wf_h086.combined_report.max_drawdown:.2%}"
    )
    print(
        f"H078 standalone: Sharpe {wf_h078.combined_report.sharpe:.3f}  MaxDD {wf_h078.combined_report.max_drawdown:.2%}"
    )
    print(
        f"H089 standalone: Sharpe {wf_h089.combined_report.sharpe:.3f}  MaxDD {wf_h089.combined_report.max_drawdown:.2%}"
    )

    # --- build blended returns over the OOS index ---
    idx = ret_h021.index
    rets_dict = {
        "h021": ret_h021,
        "h086": ret_h086.reindex(idx).fillna(0.0),
        "h078": ret_h078.reindex(idx).fillna(0.0),
        "h089": ret_h089.reindex(idx).fillna(0.0),
    }

    # compute inverse-vol weights (lagged within the WF OOS returns themselves)
    # The vol is computed on the combined OOS returns; weights are updated quarterly
    weight_df = quarterly_rebalance_weights(rets_dict, idx)

    ret_df = pd.DataFrame(rets_dict).reindex(idx).fillna(0.0)
    blended_rets = (weight_df * ret_df).sum(axis=1)

    lo, hi = bootstrap_sharpe_ci(blended_rets, n_boot=5000)
    bl_sharpe = float(sharpe(blended_rets))
    bl_cagr = float((1 + blended_rets).prod() ** (252 / len(blended_rets)) - 1)
    cum = (1 + blended_rets).cumprod()
    bl_maxdd = float((cum / cum.cummax() - 1).min())
    bl_ntrades = int(
        wf_h021.combined_report.n_trades
        + wf_h086.combined_report.n_trades
        + wf_h078.combined_report.n_trades
        + wf_h089.combined_report.n_trades
    )

    print("\nBlended result:")
    print(f"  Sharpe {bl_sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {bl_cagr:.2%}  MaxDD {bl_maxdd:.2%}  N_trades {bl_ntrades}")

    # --- average weights ---
    avg_w = weight_df.mean()
    print("\nAverage weights (OOS):")
    for k, v in avg_w.items():
        print(f"  {k}: {v:.1%}")

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
        hypothesis_id="344",
        name="vol-targeted-multi-sleeve",
        strategy_class="VolTargetedMultiSleeve",
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
            "Runner-level ensemble: H021 + H086 + H078 + H089 blended via inverse-vol weights. "
            "Weights = 1/sigma (252d trailing vol), normalized; quarterly rebalance only "
            "(first trading day of Jan/Apr/Jul/Oct). Risk-parity applied at strategy level. "
            "Vol computed on OOS returns; no lookahead (weights use realized, not prospective vol)."
        ),
        canonical_params={
            "vol_window": VOL_WINDOW,
            "rebalance_months": [1, 4, 7, 10],
            "sleeves": ["H021", "H086", "H078", "H089"],
        },
    )


if __name__ == "__main__":
    main()
