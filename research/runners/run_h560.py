"""H560: Concentrated Pool MDR.

Tests whether a reduced pool (top 20 or top 30 strategies by base Sharpe) produces
better leveraged-Calmar MDR results than the full 55-strategy pool (H558: Cal_lev 1.802).

Based on H516 finding: focused smaller pool > full pool for MDR optimization quality.

Run: uv run python scripts/run_h560.py
Expected runtime: ~3-4 hours (three pool sizes x two objectives).
"""

from __future__ import annotations

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

# H556 leverage params (best found)
LEV_TARGET_VOL = 0.065
LEV_MAX = 2.0
LEV_LOOKBACK = 20

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


def apply_vol_leverage(r_arr):
    """Apply H556's vol-targeting leverage to a return array."""
    r = pd.Series(r_arr)
    rv = r.rolling(LEV_LOOKBACK, min_periods=LEV_LOOKBACK).std().shift(1) * np.sqrt(TRADING_DAYS)
    lev = (LEV_TARGET_VOL / rv).clip(upper=LEV_MAX)
    lev = lev.where(rv.notna() & (rv > 0), 1.0)
    borrow = (lev - 1.0).clip(lower=0.0) * BORROW_COST_ANNUAL / TRADING_DAYS
    return (lev * r - borrow).values


def portfolio_metrics(r_arr):
    """Return (sharpe, calmar, cagr, maxdd) for a return array."""
    ann_mean = r_arr.mean() * TRADING_DAYS
    ann_std = r_arr.std() * np.sqrt(TRADING_DAYS)
    sh = ann_mean / ann_std if ann_std > 0 else 0.0
    cal_cagr = (1 + r_arr).prod() ** (TRADING_DAYS / len(r_arr)) - 1
    cum = np.cumprod(1 + r_arr)
    dd = (cum / np.maximum.accumulate(cum) - 1).min()
    calmar = cal_cagr / abs(dd) if dd < 0 else 0.0
    return sh, calmar, cal_cagr, dd


def run_pool(mat_sub, names_sub, label):
    """Run MDR warm-start + leveraged-Calmar objective on a subpool."""
    n = mat_sub.shape[1]
    print(f"\n{SEP}")
    print(f"POOL: {label} ({n} strategies)")
    print(SEP)
    print(f"  Strategies: {names_sub}")

    # MDR warm-start for this pool
    w0 = mdr_warm_start(mat_sub, n)

    # MDR solution (for reference)
    cov = np.cov(mat_sub.T) * TRADING_DAYS
    vols = np.sqrt(np.diag(cov))

    def neg_dr(w):
        pv = w @ cov @ w
        return 0.0 if pv <= 0 else -(np.dot(w, vols) / np.sqrt(pv))

    res_mdr = minimize(
        neg_dr,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w_mdr = res_mdr.x
    r_mdr = mat_sub @ w_mdr
    sh_m, cal_m, cg_m, dd_m = portfolio_metrics(r_mdr)
    r_mdr_lev = apply_vol_leverage(r_mdr)
    sh_ml, cal_ml, cg_ml, dd_ml = portfolio_metrics(r_mdr_lev)
    print(
        f"  [MDR] base: Sh={sh_m:.3f} Cal={cal_m:.3f} CAGR={cg_m:.2%} DD={dd_m:.2%} | "
        f"lev: Sh={sh_ml:.3f} Cal={cal_ml:.3f} CAGR={cg_ml:.2%} DD={dd_ml:.2%}"
    )

    # Leveraged-Calmar objective
    def neg_leveraged_calmar(w):
        r_base = mat_sub @ w
        r_lev = apply_vol_leverage(r_base)
        cg = (1 + r_lev).prod() ** (TRADING_DAYS / len(r_lev)) - 1
        cum = np.cumprod(1 + r_lev)
        dd = (cum / np.maximum.accumulate(cum) - 1).min()
        return -(cg / abs(dd)) if dd < 0 else 0.0

    res_cal = minimize(
        neg_leveraged_calmar,
        w0,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    w_cal = res_cal.x
    r_cal = mat_sub @ w_cal
    sh_c, cal_c, cg_c, dd_c = portfolio_metrics(r_cal)
    r_cal_lev = apply_vol_leverage(r_cal)
    sh_cl, cal_cl, cg_cl, dd_cl = portfolio_metrics(r_cal_lev)
    print(
        f"  [LevCalmar] base: Sh={sh_c:.3f} Cal={cal_c:.3f} CAGR={cg_c:.2%} DD={dd_c:.2%} | "
        f"lev: Sh={sh_cl:.3f} Cal={cal_cl:.3f} CAGR={cg_cl:.2%} DD={dd_cl:.2%}"
    )

    top_w_mdr = {names_sub[i]: float(v) for i, v in enumerate(w_mdr) if v > 0.005}
    top_w_cal = {names_sub[i]: float(v) for i, v in enumerate(w_cal) if v > 0.005}
    print(f"  MDR weights: {top_w_mdr}")
    print(f"  LevCalmar weights: {top_w_cal}")

    return {
        "mdr": (w_mdr, r_mdr, sh_m, cal_m, cg_m, dd_m, sh_ml, cal_ml, cg_ml, dd_ml, top_w_mdr),
        "lev_calmar": (
            w_cal,
            r_cal,
            sh_c,
            cal_c,
            cg_c,
            dd_c,
            sh_cl,
            cal_cl,
            cg_cl,
            dd_cl,
            top_w_cal,
        ),
    }


def main():
    print(f"H560: Concentrated Pool MDR\n{SEP}")
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
    n_full = mat.shape[1]
    print(f"\nFull pool: {n_full} strategies")

    # Rank strategies by base Sharpe
    base_sharpes = {nm: sharpe(clean[nm].fillna(0.0)) for nm in strategy_names}
    ranked = sorted(strategy_names, key=lambda k: -base_sharpes[k])
    print("\nTop 30 strategies by base Sharpe:")
    for i, nm in enumerate(ranked[:30]):
        print(f"  {i + 1:>2}. {nm}: {base_sharpes[nm]:.3f}")

    top20_names = ranked[:20]
    top30_names = ranked[:30]

    # Build subpool matrices
    mat_top30 = clean[top30_names].fillna(0.0).values
    mat_top20 = clean[top20_names].fillna(0.0).values

    # Run all three pool sizes
    results_full = run_pool(mat, strategy_names, "Full pool (55)")
    results_top30 = run_pool(mat_top30, top30_names, "Top-30 pool")
    results_top20 = run_pool(mat_top20, top20_names, "Top-20 pool")

    # Summary comparison
    print(f"\n{SEP}")
    print("CROSS-POOL COMPARISON (leveraged-Calmar objective)")
    print(SEP)
    print(
        f"  {'Pool':<23} {'Sh_base':>8} {'Cal_base':>9} {'Sh_lev':>8} {'Cal_lev':>9} {'CAGR_lev':>10} {'DD_lev':>8}"
    )
    print("-" * 80)
    for pool_label, res in [
        ("Full-55", results_full),
        ("Top-30", results_top30),
        ("Top-20", results_top20),
    ]:
        _, r, sh, cal, cg, dd, shl, call, cgl, ddl, _ = res["lev_calmar"]
        print(
            f"  {pool_label:<23} {sh:>8.3f} {cal:>9.3f} {shl:>8.3f} {call:>9.3f} {cgl * 100:>9.2f}%  {ddl * 100:>7.2f}%"
        )

    # Select best pool by Cal_lev
    pool_results = [
        ("Full-55", results_full["lev_calmar"], mat, strategy_names),
        ("Top-30", results_top30["lev_calmar"], mat_top30, top30_names),
        ("Top-20", results_top20["lev_calmar"], mat_top20, top20_names),
    ]
    pool_results.sort(key=lambda x: -x[1][7])  # sort by cal_lev (index 7)
    best_pool_label, best_res, best_mat, best_names = pool_results[0]
    (
        _,
        best_r,
        best_sh,
        best_cal,
        best_cg,
        best_dd,
        best_shl,
        best_call,
        best_cgl,
        best_ddl,
        best_top_w,
    ) = best_res
    print(f"\nBest pool (max Cal_lev = {best_call:.3f}): {best_pool_label}")

    best_port_lev = pd.Series(apply_vol_leverage(best_r), index=clean.index)

    print(f"\n{SEP}")
    print(f"FINAL ({best_pool_label} + leveraged-Calmar)")
    print(SEP)
    lo, hi = bootstrap_sharpe_ci(best_port_lev, n_boot=5000)
    print(
        f"Leveraged: Sharpe {best_shl:.3f}  CI [{lo:.3f}, {hi:.3f}]  "
        f"CAGR {best_cgl:.2%}  MaxDD {best_ddl:.2%}  Calmar {best_call:.3f}"
    )
    print(f"Top weights: {best_top_w}")

    print("\nPer-fold Sharpe (leveraged):")
    fold_sharpes = []
    for i, fold in enumerate(folds_aa):
        mask = (best_port_lev.index >= fold.test_start) & (best_port_lev.index < fold.test_end)
        fr = best_port_lev[mask]
        fs = sharpe(fr) if len(fr) > 10 else 0.0
        fold_sharpes.append(float(fs))
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_aa)
    corr = best_port_lev.corr(wh.combined_returns.reindex(best_port_lev.index))
    print(f"\nH001 correlation: {corr:+.4f}")

    decision = "KEEP" if best_shl > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    # Comparison string
    comp_str = " | ".join(
        f"{pl}: Sh_lev={res[6]:.3f}/Cal_lev={res[7]:.3f}/CAGR={res[8]:.2%}/DD={res[9]:.2%}"
        for pl, res, _, _ in pool_results
    )

    write_summary(
        hypothesis_id="560",
        name="concentrated-pool-mdr",
        strategy_class="ConcentratedPoolMDR",
        family="Set8 Batch3 Ensemble",
        sharpe=best_shl,
        ci_low=lo,
        ci_high=hi,
        cagr=best_cgl,
        max_drawdown=best_ddl,
        n_trades=0,
        h001_corr=float(corr),
        fold_sharpes=fold_sharpes,
        decision=decision,
        notes=(
            f"Best pool: {best_pool_label} (selected by max Cal_lev). "
            f"Leveraged-Calmar objective (H558 winner) applied to reduced pools. "
            f"Pool comparison: {comp_str}. "
            f"H558 baseline: Cal_lev=1.802/CAGR=11.89%/Sh_lev=1.697. "
            f"H516 hypothesis: smaller focused pool > full pool for MDR."
        ),
        canonical_params={
            "best_pool": best_pool_label,
            "objective": "leveraged-calmar",
            "leverage_target_vol": LEV_TARGET_VOL,
            "leverage_max": LEV_MAX,
            "leverage_lookback": LEV_LOOKBACK,
            "top_weights": best_top_w,
            "pool_sizes_tested": [20, 30, 55],
        },
    )


if __name__ == "__main__":
    main()
