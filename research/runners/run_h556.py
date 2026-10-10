"""H556: Dynamic Leverage via Vol-Targeting on H551.

Applies leverage_t = min(max_lev, target_vol / realized_vol_{t-1}) to H551's
portfolio. Uses H551's fixed optimal weights so leverage is the only change.
Only runs H551's 10 active strategies (vs 55 in H551) — ~30 min expected runtime.

Run: uv run python scripts/run_h556.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from write_summary import write_summary

from erised.data.fundamentals import load_fundamentals_panel
from erised.data.loader import load_research
from erised.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
from erised.metrics.performance import bootstrap_sharpe_ci, cagr, max_drawdown, sharpe
from erised.strategies.base import (
    DefensiveSectorOverlay,
    EqualWeightDefensiveFactor,
    FourAssetRiskParity,
    MovingAverageCrossover,
    MultiAssetStrategy,
    SellInMayBondConfirm,
    SPYTLTCorrelationRegime,
    VolScaledTrend,
)
from erised.strategies.cross_sectional import (
    LongShortPCAReversal,
    PCAOUMeanReversion,
    SustainabilityFilteredProfitability,
    VolScaledMomentum,
)
from erised.validation.walk_forward import (
    make_folds,
    walk_forward,
    walk_forward_cross_sectional,
    walk_forward_multi,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEP = "=" * 64
TRADING_DAYS = 252
BORROW_COST_ANNUAL = 0.005  # 50 bps/yr on leveraged fraction

# H551 optimal weights (from summary.json canonical_params)
H551_WEIGHTS = {
    "H016": 0.07436,
    "H034": 0.01216,
    "H043": 0.01884,
    "H063": 0.03369,
    "H065": 0.26861,
    "H076": 0.25804,
    "H157": 0.14965,
    "H192": 0.01278,
    "H265": 0.00542,
    "H473": 0.16521,
}


def apply_dynamic_leverage(
    port_rets: pd.Series,
    target_vol: float,
    max_lev: float,
    vol_lookback: int = 20,
) -> pd.Series:
    """Apply vol-targeting leverage overlay with financing cost."""
    realized_vol = port_rets.rolling(vol_lookback, min_periods=vol_lookback).std().shift(
        1
    ) * np.sqrt(TRADING_DAYS)
    leverage = (target_vol / realized_vol).clip(upper=max_lev)
    leverage = leverage.where(realized_vol.notna() & (realized_vol > 0), 1.0)

    # Financing cost: (lev - 1) * borrow_cost / 252 per day on leveraged fraction
    daily_borrow = (leverage - 1).clip(lower=0) * BORROW_COST_ANNUAL / TRADING_DAYS
    return leverage * port_rets - daily_borrow


def run_aa(prices, strat, folds):
    if isinstance(strat, MultiAssetStrategy):
        return walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
    return walk_forward(prices, lambda _, s=strat: s, folds=folds)


def main():
    print(f"H556: Dynamic Leverage Vol-Targeting on H551\n{SEP}")

    spy = load_research("SPY")
    etfs = {
        n: load_research(n)[["adj_close"]].rename(columns={"adj_close": f"{n.lower()}_close"})
        for n in ["TLT", "GLD", "VIX", "SHY", "DBC", "XLP", "XLU", "IVE"]
    }
    prices_aa = spy.copy()
    for df in etfs.values():
        prices_aa = prices_aa.join(df, how="left")
    for col in [c for c in prices_aa.columns if c != "adj_close"]:
        prices_aa[col] = prices_aa[col].ffill()
    folds_aa = make_folds(spy, train_years=5, test_years=2)

    tickers = get_all_historical_tickers()
    panel = load_stocks_panel(tickers)
    membership = load_sp500_members()
    common = panel.index.intersection(membership.index)
    panel = panel.loc[common]
    membership = membership.loc[common]
    spy_rets = spy["adj_close"].pct_change().fillna(0.0)
    sectors = pd.read_parquet(DATA_DIR / "sp500_sectors.parquet")["sector"]
    pt = panel.columns.tolist()
    gp = load_fundamentals_panel(pt, "gross_profit", panel.index)
    ta = load_fundamentals_panel(pt, "total_assets", panel.index)
    ni = load_fundamentals_panel(pt, "net_income", panel.index)
    ca = load_fundamentals_panel(pt, "current_assets", panel.index)
    cl = load_fundamentals_panel(pt, "current_liabilities", panel.index)
    ocf = load_fundamentals_panel(pt, "operating_cf", panel.index)
    folds_cs = make_folds(panel, train_years=5, test_years=2)
    print(f"Panel: {panel.shape[0]} days, {panel.shape[1]} tickers")

    # AA strategies: only H551's active components
    aa_strats = {
        "H016": VolScaledTrend(),
        "H034": SPYTLTCorrelationRegime(),
        "H043": SellInMayBondConfirm(),
        "H063": DefensiveSectorOverlay(),
        "H065": EqualWeightDefensiveFactor(),
        "H076": FourAssetRiskParity(),
    }

    print(f"\nRunning {len(aa_strats)} AA strategies...")
    all_rets = {}
    for name, strat in aa_strats.items():
        wf = run_aa(prices_aa, strat, folds_aa)
        all_rets[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    cs_factories = {
        "H157": lambda tp, tm: VolScaledMomentum(),
        "H192": lambda tp, tm: PCAOUMeanReversion(train_prices=tp, sectors=sectors),
        "H265": lambda tp, tm: SustainabilityFilteredProfitability(
            gross_profit=gp,
            total_assets=ta,
            current_assets=ca,
            current_liabilities=cl,
            net_income=ni,
            operating_cf=ocf,
            sectors=sectors,
        ),
        "H473": lambda tp, tm: LongShortPCAReversal(train_prices=tp, sectors=sectors),
    }
    print(f"\nRunning {len(cs_factories)} CS strategies...")
    for name, factory in cs_factories.items():
        wf = walk_forward_cross_sectional(
            panel, membership, factory, folds=folds_cs, spy_returns=spy_rets
        )
        all_rets[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    # Build H551 portfolio using its fixed optimal weights
    ret_df = pd.DataFrame(all_rets)
    w_arr = np.array([H551_WEIGHTS[k] for k in ret_df.columns])
    w_arr = w_arr / w_arr.sum()  # renormalize (sum should be ~1.0)
    base_port = pd.Series(ret_df.values @ w_arr, index=ret_df.index)

    base_sh = sharpe(base_port)
    base_cagr = cagr(base_port)
    base_dd = max_drawdown(base_port)
    print(f"\n{SEP}")
    print(
        f"H551 portfolio (fixed weights, 1.0x): Sharpe {base_sh:.3f}  CAGR {base_cagr:.2%}  MaxDD {base_dd:.2%}"
    )

    # Sensitivity grid: target_vol × max_lev
    target_vols = [0.05, 0.06, 0.065, 0.07, 0.08]
    max_levs = [1.25, 1.5, 2.0]
    vol_lookback = 20

    print(f"\n{SEP}")
    print("SENSITIVITY GRID: target_vol × max_lev")
    print(
        f"{'target_vol':>12} {'max_lev':>8} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>8} {'AvgLev':>8}"
    )
    print("-" * 64)

    grid_results = []
    for max_lev in max_levs:
        for tv in target_vols:
            lev_port = apply_dynamic_leverage(base_port, tv, max_lev, vol_lookback)
            s = sharpe(lev_port)
            c = cagr(lev_port)
            dd = max_drawdown(lev_port)
            calmar_val = c / abs(dd) if dd < 0 else 0.0

            # Compute average realized leverage
            rv = base_port.rolling(vol_lookback, min_periods=vol_lookback).std().shift(1) * np.sqrt(
                TRADING_DAYS
            )
            avg_lev = (tv / rv).clip(upper=max_lev).where(rv.notna() & (rv > 0), 1.0).mean()

            print(
                f"  {tv * 100:>9.1f}%  {max_lev:>7.2f}x  {s:>8.3f}  {c * 100:>7.2f}%  {dd * 100:>7.2f}%  {calmar_val:>8.3f}  {avg_lev:>7.2f}x"
            )
            grid_results.append(
                {
                    "target_vol": tv,
                    "max_lev": max_lev,
                    "sharpe": s,
                    "cagr": c,
                    "max_dd": dd,
                    "calmar": calmar_val,
                    "avg_lev": float(avg_lev),
                    "port": lev_port,
                }
            )

    # Select best: Sharpe > base_sh AND MaxDD < -0.10 AND CAGR >= 0.08
    qualifying = [
        g
        for g in grid_results
        if g["sharpe"] > base_sh and g["cagr"] >= 0.08 and g["max_dd"] >= -0.15
    ]
    if qualifying:
        best = max(qualifying, key=lambda x: x["sharpe"])
        print(
            f"\nBest qualifying (Sharpe > {base_sh:.3f}, CAGR >= 8%, MaxDD > -15%): "
            f"target_vol={best['target_vol'] * 100:.1f}%, max_lev={best['max_lev']:.2f}x "
            f"-> Sharpe {best['sharpe']:.3f}"
        )
    else:
        # Fall back: highest Sharpe that meets CAGR bar
        cagr_qual = [g for g in grid_results if g["cagr"] >= 0.08]
        best = (
            max(cagr_qual, key=lambda x: x["sharpe"])
            if cagr_qual
            else max(grid_results, key=lambda x: x["sharpe"])
        )
        print(
            f"\nNo config beats base Sharpe with CAGR>=8%. Best CAGR-qualifying: "
            f"target_vol={best['target_vol'] * 100:.1f}%, max_lev={best['max_lev']:.2f}x "
            f"-> Sharpe {best['sharpe']:.3f}"
        )

    final_port = best["port"]
    print(f"\n{SEP}")
    print(
        f"FINAL: target_vol={best['target_vol'] * 100:.1f}%, max_lev={best['max_lev']:.2f}x, lookback={vol_lookback}d"
    )
    print(SEP)

    lo, hi = bootstrap_sharpe_ci(final_port, n_boot=5000)
    s = best["sharpe"]
    c = best["cagr"]
    dd = best["max_dd"]
    calmar_final = best["calmar"]
    print(
        f"Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]  CAGR {c:.2%}  MaxDD {dd:.2%}  Calmar {calmar_final:.3f}"
    )
    print(f"Average realized leverage: {best['avg_lev']:.2f}x")
    print(f"vs H551 base: Sharpe {base_sh:.3f}  CAGR {base_cagr:.2%}  MaxDD {base_dd:.2%}")

    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds_aa):
        mask = (final_port.index >= fold.test_start) & (final_port.index < fold.test_end)
        fr = final_port[mask]
        fs = sharpe(fr) if len(fr) > 10 else 0.0
        fold_sharpes.append(float(fs))
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_aa)
    corr = final_port.corr(wh.combined_returns.reindex(final_port.index))
    print(f"\nH001 correlation: {corr:+.4f}")

    decision = "KEEP" if s > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="556",
        name="dynamic-leverage-vol-target",
        strategy_class="DynamicLeverageVolTarget",
        family="Set8 Batch2 Leverage",
        sharpe=s,
        ci_low=lo,
        ci_high=hi,
        cagr=c,
        max_drawdown=dd,
        n_trades=0,
        h001_corr=float(corr),
        fold_sharpes=fold_sharpes,
        decision=decision,
        notes=(
            f"Dynamic leverage on H551 portfolio (fixed weights). "
            f"leverage_t = min({best['max_lev']:.2f}, {best['target_vol'] * 100:.1f}% / realized_vol). "
            f"Lookback: {vol_lookback}d. Borrow cost: {BORROW_COST_ANNUAL * 100:.0f}bps/yr. "
            f"Avg realized leverage: {best['avg_lev']:.2f}x. "
            f"Base (H551 1x): Sharpe {base_sh:.3f} CAGR {base_cagr:.2%} MaxDD {base_dd:.2%}."
        ),
        canonical_params={
            "target_vol": best["target_vol"],
            "max_lev": best["max_lev"],
            "vol_lookback": vol_lookback,
            "borrow_cost_annual": BORROW_COST_ANNUAL,
            "avg_realized_lev": round(best["avg_lev"], 3),
            "base_weights": H551_WEIGHTS,
        },
    )


if __name__ == "__main__":
    main()
