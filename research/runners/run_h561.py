"""H561: Leverage Parameter Grid Search.

Tests 27 combinations of (target_vol, cap, lookback) for the lev-Sharpe SLSQP
to determine if H559's fixed params (6.5%/2.0x/20d) are globally optimal.

Run: uv run python scripts/run_h561.py
Expected runtime: ~4-5 hours (data load + 27 SLSQP runs).
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from write_summary import write_summary

from erised.data.fundamentals import load_fundamentals_panel
from erised.data.loader import load_research
from erised.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies.base import (
    AbsoluteMomentumStrategy,
    CarryMomentumSPYTLT,
    CorrelationRiskParityHybrid,
    CrisisProtocolRiskParity,
    DefensiveEquityHedge,
    DefensiveSectorOverlay,
    EqualWeightDefensiveFactor,
    FourAssetRiskParity,
    GoldFearGauge,
    H001H021DirectBlend,
    H043H021RegimeBlend,
    HalloweenMomentumConfirm,
    JanuaryBarometer,
    LowVolTargetPortfolio,
    MomentumCarryBlend,
    MonthlyRebalancingRiskParity,
    MovingAverageCrossover,
    MultiAssetDynamicVolTarget,
    MultiAssetStrategy,
    RatchetDrawdownProtection,
    RealizedVsImpliedVol,
    ReversionEntryRiskParity,
    RiskAdjustedMomentum,
    RiskParity,
    RiskParityMomentumFilter,
    RiskParityYieldCurveOverlay,
    SeasonalityTrendCombo,
    SeasonalRiskParity,
    SellInMay,
    SellInMayBondConfirm,
    SPYGLDRotation,
    SPYTLTCorrelationRegime,
    SPYTLTDualMomentum,
    ThreeAssetRiskParity,
    TimeSeriesMomentum,
    VixRiskParityEnhanced,
    VolScaledTrend,
    VrpCalendarHybrid,
)
from erised.strategies.cross_sectional import (
    BettingAgainstBeta,
    CostAwareMomentumLongHold,
    InvestmentEfficiencyRatio,
    KnowledgeReturnProxy,
    LongShortPCAReversal,
    LongShortResidualReversal,
    LongShortReversal,
    LongShortSectorReversal,
    LowCoverageMomentumProxy,
    MultiSignalComposite,
    PCAOUMeanReversion,
    PCAResidualReversal,
    ResidualOUMeanReversion,
    ResidualWeeklyReversal,
    SustainabilityFilteredProfitability,
    TailRiskComposite,
    VolScaledMomentum,
    WeeklyReversalSectorNeutral,
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
BORROW_COST_ANNUAL = 0.005

# H559 baseline
H559_SH_LEV = 1.879
H559_PARAMS = (0.065, 2.0, 20)

AA_POOL = {
    "H001": MovingAverageCrossover(50, 200),
    "H007": TimeSeriesMomentum(),
    "H008": AbsoluteMomentumStrategy(),
    "H016": VolScaledTrend(),
    "H018": SPYTLTDualMomentum(),
    "H019": SPYGLDRotation(),
    "H021": RiskParity(),
    "H022": DefensiveEquityHedge(),
    "H023": SellInMay(),
    "H025": JanuaryBarometer(),
    "H027": SeasonalityTrendCombo(),
    "H031": RealizedVsImpliedVol(),
    "H034": SPYTLTCorrelationRegime(),
    "H035": GoldFearGauge(),
    "H043": SellInMayBondConfirm(),
    "H046": HalloweenMomentumConfirm(),
    "H051": ThreeAssetRiskParity(),
    "H060": CarryMomentumSPYTLT(),
    "H063": DefensiveSectorOverlay(),
    "H065": EqualWeightDefensiveFactor(),
    "H066": VixRiskParityEnhanced(),
    "H068": VrpCalendarHybrid(),
    "H070": MultiAssetDynamicVolTarget(),
    "H076": FourAssetRiskParity(),
    "H077": RiskParityMomentumFilter(),
    "H078": SeasonalRiskParity(),
    "H079": CrisisProtocolRiskParity(),
    "H080": RiskParityYieldCurveOverlay(),
    "H082": RiskAdjustedMomentum(),
    "H084": MomentumCarryBlend(),
    "H086": LowVolTargetPortfolio(),
    "H087": RatchetDrawdownProtection(),
    "H088": H001H021DirectBlend(),
    "H089": H043H021RegimeBlend(),
    "H090": MonthlyRebalancingRiskParity(),
    "H097": CorrelationRiskParityHybrid(),
    "H098": ReversionEntryRiskParity(),
}


def run_aa(prices, strat, folds):
    if isinstance(strat, MultiAssetStrategy):
        return walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
    return walk_forward(prices, lambda _, s=strat: s, folds=folds)


def mdr_warm_start(mat, n):
    """MDR solution used as warm-start for all objectives."""
    cov = np.cov(mat.T) * TRADING_DAYS
    vols = np.sqrt(np.diag(cov))

    def neg_dr(w):
        pv = w @ cov @ w
        return 0.0 if pv <= 0 else -(np.dot(w, vols) / np.sqrt(pv))

    res = minimize(
        neg_dr,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return res.x


def apply_vol_leverage_params(r_arr, target_vol, cap, lookback):
    """Apply vol-targeting with configurable params."""
    r = pd.Series(r_arr)
    rv = r.rolling(lookback, min_periods=lookback).std().shift(1) * np.sqrt(TRADING_DAYS)
    lev = (target_vol / rv).clip(upper=cap)
    lev = lev.where(rv.notna() & (rv > 0), 1.0)
    borrow = (lev - 1.0).clip(lower=0.0) * BORROW_COST_ANNUAL / TRADING_DAYS
    return (lev * r - borrow).values


def portfolio_metrics(r_arr):
    """Return (sharpe, calmar, cagr, maxdd, sortino) for a return array."""
    ann_mean = r_arr.mean() * TRADING_DAYS
    ann_std = r_arr.std() * np.sqrt(TRADING_DAYS)
    sh = ann_mean / ann_std if ann_std > 0 else 0.0
    cal_cagr = (1 + r_arr).prod() ** (TRADING_DAYS / len(r_arr)) - 1
    cum = np.cumprod(1 + r_arr)
    dd = (cum / np.maximum.accumulate(cum) - 1).min()
    calmar = cal_cagr / abs(dd) if dd < 0 else 0.0
    ds = r_arr[r_arr < 0]
    ds_std = np.sqrt((ds**2).mean()) * np.sqrt(TRADING_DAYS) if len(ds) > 0 else 1.0
    sortino = ann_mean / ds_std if ds_std > 0 else 0.0
    return sh, calmar, cal_cagr, dd, sortino


def neg_lev_sharpe_params(w, mat, target_vol, cap, lookback):
    r_base = mat @ w
    r_lev = apply_vol_leverage_params(r_base, target_vol, cap, lookback)
    ann_mean = r_lev.mean() * TRADING_DAYS
    ann_std = r_lev.std() * np.sqrt(TRADING_DAYS)
    return -(ann_mean / ann_std) if ann_std > 0 else 0.0


def run_grid(mat, w0, param_grid):
    """Run lev-Sharpe SLSQP for each (target_vol, cap, lookback) combo."""
    results = []
    for tv, cap, lb in param_grid:
        res = minimize(
            neg_lev_sharpe_params,
            w0,
            args=(mat, tv, cap, lb),
            method="SLSQP",
            bounds=[(0, 1)] * mat.shape[1],
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        w = res.x
        r_base = mat @ w
        r_lev = apply_vol_leverage_params(r_base, tv, cap, lb)
        sh_b, cal_b, cg_b, dd_b, so_b = portfolio_metrics(r_base)
        sh_l, cal_l, cg_l, dd_l, so_l = portfolio_metrics(r_lev)
        # avg leverage
        r_s = pd.Series(r_base)
        rv = r_s.rolling(lb, min_periods=lb).std().shift(1) * np.sqrt(TRADING_DAYS)
        avg_lev = float((tv / rv).clip(upper=cap).where(rv.notna() & (rv > 0), 1.0).mean())
        results.append(
            {
                "target_vol": tv,
                "cap": cap,
                "lookback": lb,
                "sh_base": sh_b,
                "sh_lev": sh_l,
                "cal_lev": cal_l,
                "cg_lev": cg_l,
                "dd_lev": dd_l,
                "so_lev": so_l,
                "avg_lev": avg_lev,
                "weights": w,
            }
        )
        print(
            f"  tv={tv:.3f} cap={cap:.1f}x lb={lb}d | "
            f"Sh_lev={sh_l:.3f} Cal_lev={cal_l:.3f} "
            f"CAGR={cg_l:.2%} DD={dd_l:.2%} avg_lev={avg_lev:.2f}x"
        )
    return results


def main():
    print(f"H561: Leverage Parameter Grid Search\n{SEP}")
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
    rev = load_fundamentals_panel(pt, "revenue", panel.index)
    folds_cs = make_folds(panel, train_years=5, test_years=2)
    print(f"Panel: {panel.shape[0]} days, {panel.shape[1]} tickers")

    print(f"\nRunning {len(AA_POOL)} AA strategies...")
    all_rets = {}
    for name, strat in AA_POOL.items():
        wf = run_aa(prices_aa, strat, folds_aa)
        all_rets[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    cs_factories = {
        "H157": lambda tp, tm: VolScaledMomentum(),
        "H175": lambda tp, tm: BettingAgainstBeta(spy_rets=spy_rets, sectors=sectors),
        "H185": lambda tp, tm: WeeklyReversalSectorNeutral(sectors=sectors),
        "H186": lambda tp, tm: ResidualWeeklyReversal(spy_rets=spy_rets, sectors=sectors),
        "H189": lambda tp, tm: ResidualOUMeanReversion(spy_rets=spy_rets, sectors=sectors),
        "H191": lambda tp, tm: PCAResidualReversal(train_prices=tp, sectors=sectors),
        "H192": lambda tp, tm: PCAOUMeanReversion(train_prices=tp, sectors=sectors),
        "H194": lambda tp, tm: LongShortReversal(sectors=sectors),
        "H199": lambda tp, tm: MultiSignalComposite(net_income=ni, sectors=sectors),
        "H254": lambda tp, tm: TailRiskComposite(sectors=sectors, spy_rets=spy_rets),
        "H265": lambda tp, tm: SustainabilityFilteredProfitability(
            gross_profit=gp,
            total_assets=ta,
            current_assets=ca,
            current_liabilities=cl,
            net_income=ni,
            operating_cf=ocf,
            sectors=sectors,
        ),
        "H268": lambda tp, tm: KnowledgeReturnProxy(
            gross_profit=gp, total_assets=ta, sectors=sectors
        ),
        "H284": lambda tp, tm: LowCoverageMomentumProxy(
            spy_prices=spy["adj_close"], sectors=sectors
        ),
        "H287": lambda tp, tm: InvestmentEfficiencyRatio(
            revenue=rev, total_assets=ta, current_assets=ca, sectors=sectors
        ),
        "H293": lambda tp, tm: CostAwareMomentumLongHold(sectors=sectors),
        "H471": lambda tp, tm: LongShortSectorReversal(sectors=sectors),
        "H472": lambda tp, tm: LongShortResidualReversal(spy_rets=spy_rets, sectors=sectors),
        "H473": lambda tp, tm: LongShortPCAReversal(train_prices=tp, sectors=sectors),
    }
    print(f"\nRunning {len(cs_factories)} CS strategies...")
    for name, factory in cs_factories.items():
        wf = walk_forward_cross_sectional(
            panel, membership, factory, folds=folds_cs, spy_returns=spy_rets
        )
        all_rets[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    strategy_names = list(all_rets.keys())
    ret_df = pd.DataFrame(all_rets)
    clean = ret_df.dropna(how="all")
    mat = clean.fillna(0.0).values
    n = mat.shape[1]
    print(f"\nPool: {n} strategies")

    print("\nComputing MDR warm-start...")
    w0 = mdr_warm_start(mat, n)

    param_grid = list(
        itertools.product(
            [0.05, 0.065, 0.08],  # target_vol
            [1.5, 2.0, 3.0],  # cap
            [10, 20, 40],  # lookback
        )
    )
    print(f"\nRunning {len(param_grid)} parameter combinations...")
    results = run_grid(mat, w0, param_grid)

    # Print sorted summary table
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "weights"} for r in results])
    df = df.sort_values("sh_lev", ascending=False)
    print(f"\n{SEP}")
    print("TOP 10 BY SH_LEV:")
    print(SEP)
    print(
        f"{'tv':>6} {'cap':>5} {'lb':>4} | {'Sh_lev':>8} {'Cal_lev':>9} "
        f"{'CAGR_lev':>10} {'DD_lev':>8} {'avg_lev':>9}"
    )
    for _, row in df.head(10).iterrows():
        print(
            f"  {row.target_vol:.3f} {row.cap:.1f}x {int(row.lookback):>3}d | "
            f"{row.sh_lev:>8.3f} {row.cal_lev:>9.3f} "
            f"{row.cg_lev * 100:>9.2f}% {row.dd_lev * 100:>7.2f}% "
            f"{row.avg_lev:>8.2f}x"
        )

    best = max(results, key=lambda r: r["sh_lev"])
    best_w = best["weights"]
    best_tv, best_cap, best_lb = best["target_vol"], best["cap"], best["lookback"]

    # Recompute leveraged returns with best params
    best_r_base = mat @ best_w
    best_r_lev = apply_vol_leverage_params(best_r_base, best_tv, best_cap, best_lb)
    best_port_lev = pd.Series(best_r_lev, index=clean.index)

    print(f"\n{SEP}")
    print(f"FINAL: Best params tv={best_tv:.3f} cap={best_cap:.1f}x lb={best_lb}d")
    print(SEP)

    lo, hi = bootstrap_sharpe_ci(best_port_lev, n_boot=5000)
    print(
        f"Leveraged: Sharpe {best['sh_lev']:.3f}  CI [{lo:.3f}, {hi:.3f}]  "
        f"CAGR {best['cg_lev']:.2%}  MaxDD {best['dd_lev']:.2%}  "
        f"Calmar {best['cal_lev']:.3f}  Sortino {best['so_lev']:.3f}"
    )
    top_w = {strategy_names[i]: float(v) for i, v in enumerate(best_w) if v > 0.005}
    print(f"Top weights: {top_w}")
    print(f"H559 baseline: tv=0.065, cap=2.0x, lb=20d, Sh_lev={H559_SH_LEV:.3f}")
    print(f"Improvement over H559: {best['sh_lev'] - H559_SH_LEV:+.3f}")

    print("\nPer-fold Sharpe (leveraged):")
    fold_sharpes = []
    for i, fold in enumerate(folds_aa):
        mask = (best_port_lev.index >= fold.test_start) & (best_port_lev.index < fold.test_end)
        fr = best_port_lev[mask]
        fs = float(sharpe(fr)) if len(fr) > 10 else 0.0
        fold_sharpes.append(fs)
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_aa)
    corr = best_port_lev.corr(wh.combined_returns.reindex(best_port_lev.index))
    print(f"\nH001 correlation: {corr:+.4f}")

    decision = "KEEP" if best["sh_lev"] > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    grid_top5 = "; ".join(
        f"tv={r['target_vol']:.3f}/cap={r['cap']:.1f}x/lb={int(r['lookback'])}d:Sh={r['sh_lev']:.3f}"
        for r in sorted(results, key=lambda x: -x["sh_lev"])[:5]
    )

    write_summary(
        hypothesis_id="561",
        name="leverage-param-grid-search",
        strategy_class="LeverageParamGridSearch",
        family="Set8 Batch4 Leverage",
        sharpe=best["sh_lev"],
        ci_low=lo,
        ci_high=hi,
        cagr=best["cg_lev"],
        max_drawdown=best["dd_lev"],
        n_trades=0,
        h001_corr=float(corr),
        fold_sharpes=fold_sharpes,
        decision=decision,
        notes=(
            f"Best params: target_vol={best_tv:.3f}, cap={best_cap:.1f}x, lookback={best_lb}d. "
            f"H559 baseline (0.065/2.0x/20d) Sh_lev={H559_SH_LEV:.3f}. "
            f"Improvement over H559: {best['sh_lev'] - H559_SH_LEV:+.3f}. "
            f"Top-5 by Sh_lev: {grid_top5}. "
            f"27 combinations tested."
        ),
        canonical_params={
            "best_target_vol": best_tv,
            "best_cap": best_cap,
            "best_lookback": best_lb,
            "h559_baseline_sh_lev": H559_SH_LEV,
            "n_combinations": 27,
        },
    )


if __name__ == "__main__":
    main()
