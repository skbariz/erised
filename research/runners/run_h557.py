"""H557: Ratchet-Conditional Leverage on H551.

Tests drawdown-conditional leverage as an alternative to H556's vol-targeting.
leverage_t = 1.0 + (max_lev - 1.0) * max(0, 1 - dd_{t-1} / dd_limit)
Drawdown tracked on base (unleveraged) H551 portfolio to avoid feedback circularity.
Uses same 10 active H551 components and fixed weights as H556.

Run: uv run python scripts/run_h557.py
Expected runtime: ~30 min.
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
BORROW_COST_ANNUAL = 0.005  # 50 bps/yr

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


def apply_ratchet_leverage(
    base_rets: pd.Series,
    max_lev: float,
    dd_limit: float,
) -> pd.Series:
    """Drawdown-conditional leverage. Tracks dd on base (unleveraged) portfolio."""
    cum = (1.0 + base_rets.fillna(0.0)).cumprod()
    peak = cum.expanding(min_periods=1).max()
    dd = (cum / peak - 1.0).fillna(0.0)

    # Yesterday's drawdown determines today's leverage
    ratchet = (1.0 + dd.shift(1).fillna(0.0) / dd_limit).clip(lower=0.0, upper=1.0)
    leverage = 1.0 + (max_lev - 1.0) * ratchet

    daily_borrow = (leverage - 1.0).clip(lower=0.0) * BORROW_COST_ANNUAL / TRADING_DAYS
    return leverage * base_rets - daily_borrow


def run_aa(prices, strat, folds):
    if isinstance(strat, MultiAssetStrategy):
        return walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
    return walk_forward(prices, lambda _, s=strat: s, folds=folds)


def main():
    print(f"H557: Ratchet-Conditional Leverage on H551\n{SEP}")

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

    ret_df = pd.DataFrame(all_rets)
    w_arr = np.array([H551_WEIGHTS[k] for k in ret_df.columns])
    w_arr = w_arr / w_arr.sum()
    base_port = pd.Series(ret_df.values @ w_arr, index=ret_df.index)

    base_sh = sharpe(base_port)
    base_cagr = cagr(base_port)
    base_dd = max_drawdown(base_port)
    print(f"\n{SEP}")
    print(f"H551 base (1.0x): Sharpe {base_sh:.3f}  CAGR {base_cagr:.2%}  MaxDD {base_dd:.2%}")
    print("H556 reference (vol-target 6.5%/2.0x): Sharpe 1.790  CAGR 9.61%  MaxDD -5.79%")

    dd_limits = [0.03, 0.05, 0.07, 0.10, 0.15]
    max_levs = [1.5, 2.0]

    print(f"\n{SEP}")
    print("SENSITIVITY GRID: max_lev x dd_limit")
    print(
        f"{'max_lev':>8} {'dd_limit':>9} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>8} {'AvgLev':>8}"
    )
    print("-" * 64)

    grid_results = []
    for max_lev in max_levs:
        for ddl in dd_limits:
            lev_port = apply_ratchet_leverage(base_port, max_lev, ddl)
            s = sharpe(lev_port)
            c = cagr(lev_port)
            dd = max_drawdown(lev_port)
            calmar_val = c / abs(dd) if dd < 0 else 0.0

            cum = (1.0 + base_port.fillna(0.0)).cumprod()
            peak = cum.expanding(min_periods=1).max()
            base_dd_series = (cum / peak - 1.0).fillna(0.0)
            ratchet = (1.0 + base_dd_series.shift(1).fillna(0.0) / ddl).clip(0.0, 1.0)
            avg_lev = (1.0 + (max_lev - 1.0) * ratchet).mean()

            print(
                f"  {max_lev:.2f}x   dd={ddl * 100:>4.0f}%  {s:>8.3f}  {c * 100:>7.2f}%"
                f"  {dd * 100:>7.2f}%  {calmar_val:>8.3f}  {avg_lev:>7.2f}x"
            )
            grid_results.append(
                {
                    "max_lev": max_lev,
                    "dd_limit": ddl,
                    "sharpe": s,
                    "cagr": c,
                    "max_dd": dd,
                    "calmar": calmar_val,
                    "avg_lev": float(avg_lev),
                    "port": lev_port,
                }
            )

    # Best: Sharpe > base_sh AND CAGR >= 8% AND MaxDD > -15%
    qualifying = [
        g
        for g in grid_results
        if g["sharpe"] > base_sh and g["cagr"] >= 0.08 and g["max_dd"] >= -0.15
    ]
    if qualifying:
        best = max(qualifying, key=lambda x: x["sharpe"])
        print(
            f"\nBest qualifying (Sharpe>{base_sh:.3f}, CAGR>=8%, MaxDD>-15%): "
            f"max_lev={best['max_lev']:.2f}x, dd_limit={best['dd_limit'] * 100:.0f}% "
            f"-> Sharpe {best['sharpe']:.3f}"
        )
    else:
        cagr_qual = [g for g in grid_results if g["cagr"] >= 0.08]
        best = (
            max(cagr_qual, key=lambda x: x["sharpe"])
            if cagr_qual
            else max(grid_results, key=lambda x: x["sharpe"])
        )
        print(
            f"\nNo config beats base+CAGR>=8%. Best CAGR-qualifying: "
            f"max_lev={best['max_lev']:.2f}x, dd_limit={best['dd_limit'] * 100:.0f}% "
            f"-> Sharpe {best['sharpe']:.3f}"
        )

    final_port = best["port"]
    print(f"\n{SEP}")
    print(f"FINAL: max_lev={best['max_lev']:.2f}x, dd_limit={best['dd_limit'] * 100:.0f}%")
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
    print(f"vs H551 base:  Sharpe {base_sh:.3f}  CAGR {base_cagr:.2%}  MaxDD {base_dd:.2%}")
    print("vs H556 (vol): Sharpe 1.790  CAGR 9.61%  MaxDD -5.79%  Calmar 1.660")

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
        hypothesis_id="557",
        name="ratchet-leverage",
        strategy_class="RatchetLeverage",
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
            f"Ratchet-conditional leverage on H551 portfolio (fixed weights). "
            f"leverage_t = 1 + ({best['max_lev']:.2f}-1) * max(0, 1-dd/{best['dd_limit'] * 100:.0f}%). "
            f"Drawdown tracked on base (unleveraged) portfolio. "
            f"Avg realized leverage: {best['avg_lev']:.2f}x. "
            f"Borrow cost: {BORROW_COST_ANNUAL * 100:.0f}bps/yr. "
            f"Base (H551 1x): Sharpe {base_sh:.3f} CAGR {base_cagr:.2%} MaxDD {base_dd:.2%}. "
            f"H556 (vol-target): Sharpe 1.790 CAGR 9.61% MaxDD -5.79%."
        ),
        canonical_params={
            "max_lev": best["max_lev"],
            "dd_limit": best["dd_limit"],
            "borrow_cost_annual": BORROW_COST_ANNUAL,
            "avg_realized_lev": round(best["avg_lev"], 3),
            "base_weights": H551_WEIGHTS,
        },
    )


if __name__ == "__main__":
    main()
