"""H558: Calmar-Optimized MDR.

Tests pure Calmar and leveraged-Calmar objectives on the same 55-strategy pool as H551.
H551 tested Sharpe-dominant composites; H558 fills in the Calmar-dominant end.
Warm-starts all objectives from MDR solution (same pattern as H551).

Run: uv run python scripts/run_h558.py
Expected runtime: ~3-4 hours.
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


def optimize(mat, w0, objective_fn, label):
    n = mat.shape[1]
    res = minimize(
        objective_fn,
        w0,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    w = res.x
    r = mat @ w
    sh, cal, cg, dd = portfolio_metrics(r)
    r_lev = apply_vol_leverage(r)
    sh_lev, cal_lev, cg_lev, dd_lev = portfolio_metrics(r_lev)
    print(
        f"  [{label}] base: Sh={sh:.3f} Cal={cal:.3f} CAGR={cg:.2%} DD={dd:.2%} | "
        f"lev: Sh={sh_lev:.3f} Cal={cal_lev:.3f} CAGR={cg_lev:.2%} DD={dd_lev:.2%}"
    )
    return w, r, sh, cal, cg, dd, sh_lev, cal_lev, cg_lev, dd_lev


def main():
    print(f"H558: Calmar-Optimized MDR\n{SEP}")
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

    # MDR warm-start (same as H551)
    print("\nComputing MDR warm-start...")
    w0 = mdr_warm_start(mat, n)

    # H551 reference: equal composite (0.33/0.33/0.33)
    print(f"\n{SEP}")
    print("OBJECTIVE COMPARISON (base + leveraged at 6.5%/2.0x)")
    print(SEP)

    results = []

    # H551 reference weights applied to this run's returns
    h551_w = {
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
    w_ref = np.array([h551_w.get(k, 0.0) for k in strategy_names])
    w_ref = w_ref / w_ref.sum()
    r_ref = mat @ w_ref
    sh_ref, cal_ref, cg_ref, dd_ref = portfolio_metrics(r_ref)
    r_ref_lev = apply_vol_leverage(r_ref)
    sh_ref_lev, cal_ref_lev, cg_ref_lev, dd_ref_lev = portfolio_metrics(r_ref_lev)
    print(
        f"  [H551 ref] base: Sh={sh_ref:.3f} Cal={cal_ref:.3f} CAGR={cg_ref:.2%} DD={dd_ref:.2%} | "
        f"lev: Sh={sh_ref_lev:.3f} Cal={cal_ref_lev:.3f} CAGR={cg_ref_lev:.2%} DD={dd_ref_lev:.2%}"
    )
    results.append(
        (
            "H551-ref",
            w_ref,
            r_ref,
            sh_ref,
            cal_ref,
            cg_ref,
            dd_ref,
            sh_ref_lev,
            cal_ref_lev,
            cg_ref_lev,
            dd_ref_lev,
        )
    )

    # Objective 1: Pure Calmar
    def neg_calmar(w):
        r = mat @ w
        cg = (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1
        cum = np.cumprod(1 + r)
        dd = (cum / np.maximum.accumulate(cum) - 1).min()
        return -(cg / abs(dd)) if dd < 0 else 0.0

    w1, r1, sh1, cal1, cg1, dd1, sh1l, cal1l, cg1l, dd1l = optimize(
        mat, w0, neg_calmar, "Pure Calmar"
    )
    results.append(("pure-calmar", w1, r1, sh1, cal1, cg1, dd1, sh1l, cal1l, cg1l, dd1l))

    # Objective 2: Calmar-dominant composite (w_sh=0.1, w_cal=0.8, w_so=0.1)
    def neg_calmar_dominant(w):
        r = mat @ w
        ann_mean = r.mean() * TRADING_DAYS
        ann_std = r.std() * np.sqrt(TRADING_DAYS)
        sh = ann_mean / ann_std if ann_std > 0 else 0.0
        cg = (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1
        cum = np.cumprod(1 + r)
        dd = (cum / np.maximum.accumulate(cum) - 1).min()
        cal = cg / abs(dd) if dd < 0 else 0.0
        ds = r[r < 0]
        ds_val = np.sqrt((ds**2).mean()) * np.sqrt(TRADING_DAYS) if len(ds) > 0 else 1.0
        so = ann_mean / ds_val if ds_val > 0 else 0.0
        return -(0.1 * sh + 0.8 * cal + 0.1 * so)

    w2, r2, sh2, cal2, cg2, dd2, sh2l, cal2l, cg2l, dd2l = optimize(
        mat, w0, neg_calmar_dominant, "Calmar-dominant (0.1/0.8/0.1)"
    )
    results.append(("calmar-dominant", w2, r2, sh2, cal2, cg2, dd2, sh2l, cal2l, cg2l, dd2l))

    # Objective 3: Leveraged Calmar — optimize Calmar of the vol-targeted portfolio directly
    def neg_leveraged_calmar(w):
        r_base = mat @ w
        r_lev = apply_vol_leverage(r_base)
        cg = (1 + r_lev).prod() ** (TRADING_DAYS / len(r_lev)) - 1
        cum = np.cumprod(1 + r_lev)
        dd = (cum / np.maximum.accumulate(cum) - 1).min()
        return -(cg / abs(dd)) if dd < 0 else 0.0

    w3, r3, sh3, cal3, cg3, dd3, sh3l, cal3l, cg3l, dd3l = optimize(
        mat, w0, neg_leveraged_calmar, "Leveraged Calmar"
    )
    results.append(("leveraged-calmar", w3, r3, sh3, cal3, cg3, dd3, sh3l, cal3l, cg3l, dd3l))

    # Summary table
    print(f"\n{SEP}")
    print(f"SUMMARY (base + leveraged at {LEV_TARGET_VOL * 100:.1f}%/{LEV_MAX:.1f}x)")
    print(SEP)
    print(
        f"{'Objective':<35} {'Sh_base':>8} {'Cal_base':>9} {'Sh_lev':>8} {'Cal_lev':>9} {'CAGR_lev':>10} {'DD_lev':>8}"
    )
    print("-" * 90)
    for label, _w, _r, sh, cal, _cg, _dd, shl, call, cgl, ddl in results:
        print(
            f"  {label:<33} {sh:>8.3f} {cal:>9.3f} {shl:>8.3f} {call:>9.3f} {cgl * 100:>9.2f}%  {ddl * 100:>7.2f}%"
        )

    # Select best: maximize leveraged Sharpe × leveraged Calmar product (balanced criterion)
    scored = [
        (label, w, r, sh, cal, cg, dd, shl, call, cgl, ddl, shl * call)
        for label, w, r, sh, cal, cg, dd, shl, call, cgl, ddl in results
    ]
    scored.sort(key=lambda x: -x[-1])
    (
        best_label,
        best_w,
        best_r,
        best_sh,
        best_cal,
        best_cg,
        best_dd,
        best_shl,
        best_call,
        best_cgl,
        best_ddl,
        _,
    ) = scored[0]
    print(f"\nBest (max Sharpe_lev x Calmar_lev = {scored[0][-1]:.3f}): {best_label}")

    best_port_lev = pd.Series(apply_vol_leverage(best_r), index=clean.index)

    print(f"\n{SEP}")
    print(f"FINAL ({best_label})")
    print(SEP)
    lo, hi = bootstrap_sharpe_ci(best_port_lev, n_boot=5000)
    print(
        f"Leveraged: Sharpe {best_shl:.3f}  CI [{lo:.3f}, {hi:.3f}]  "
        f"CAGR {best_cgl:.2%}  MaxDD {best_ddl:.2%}  Calmar {best_call:.3f}"
    )
    top_w = {strategy_names[i]: float(v) for i, v in enumerate(best_w) if v > 0.005}
    print(f"Top weights: {top_w}")

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

    all_obj_str = "; ".join(
        f"{label}: Sh={sh:.3f}/Cal={cal:.3f} (lev: Sh={shl:.3f}/Cal={call:.3f}/CAGR={cgl:.2%}/DD={ddl:.2%})"
        for label, _, _, sh, cal, _, _, shl, call, cgl, ddl in results
    )

    write_summary(
        hypothesis_id="558",
        name="calmar-optimized-mdr",
        strategy_class="CalmarOptimizedMDR",
        family="Set8 Batch2 Ensemble",
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
            f"Best objective: {best_label}. Leveraged at {LEV_TARGET_VOL * 100:.1f}%/{LEV_MAX:.1f}x (H556 params). "
            f"All objectives: {all_obj_str}. "
            f"Base metrics: Sh={best_sh:.3f} Cal={best_cal:.3f} CAGR={best_cg:.2%} DD={best_dd:.2%}."
        ),
        canonical_params={
            "objective": best_label,
            "leverage_target_vol": LEV_TARGET_VOL,
            "leverage_max": LEV_MAX,
            "leverage_lookback": LEV_LOOKBACK,
            "top_weights": top_w,
            "n_strategies": n,
        },
    )


if __name__ == "__main__":
    main()
