"""H551: MultiObjectiveMDR -- composite Sharpe+Calmar+Sortino objective over H543's 55-strategy pool.
Replaces pure MDR objective with multi-objective composite to push Calmar above 0.90
while holding Sharpe >= 1.5.
Run: uv run python scripts/run_h551.py
Expected runtime: ~3-4 hours.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from write_summary import write_summary

from quant.data.fundamentals import load_fundamentals_panel
from quant.data.loader import load_research
from quant.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
from quant.metrics.performance import bootstrap_sharpe_ci, cagr, max_drawdown, sharpe
from quant.strategies.base import (
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
from quant.strategies.cross_sectional import (
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
from quant.validation.walk_forward import (
    make_folds,
    walk_forward,
    walk_forward_cross_sectional,
    walk_forward_multi,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEP = "=" * 64
TRADING_DAYS = 252

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


def composite_score(w, ret_matrix):
    r = ret_matrix @ w
    ann_mean = r.mean() * TRADING_DAYS
    ann_std = r.std() * np.sqrt(TRADING_DAYS)
    sh = ann_mean / ann_std if ann_std > 0 else 0.0
    cum = np.cumprod(1 + r)
    dd = (cum / np.maximum.accumulate(cum) - 1).min()
    cal_cagr = (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1
    calmar = cal_cagr / abs(dd) if dd < 0 else 0.0
    downside = r[r < 0]
    ds = np.sqrt((downside**2).mean()) * np.sqrt(TRADING_DAYS) if len(downside) > 0 else 1.0
    sortino = ann_mean / ds if ds > 0 else 0.0
    return sh, calmar, sortino


def multi_obj_weights(returns_df, w1=0.4, w2=0.4, w3=0.2):
    clean = returns_df.dropna(how="all")
    mat = clean.fillna(0.0).values  # fill partial NaN (strategies not yet active = 0 return)
    n = mat.shape[1]

    # Warm-start: solve smooth MDR first, use its solution as starting point.
    # Equal-weight start causes SLSQP to get stuck on non-smooth composite objective.
    cov = np.cov(mat.T) * 252
    vols = np.sqrt(np.diag(cov))

    def neg_dr(w):
        pv = w @ cov @ w
        return 0.0 if pv <= 0 else -(np.dot(w, vols) / np.sqrt(pv))

    mdr_result = minimize(
        neg_dr, np.full(n, 1.0 / n), method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w0 = mdr_result.x  # MDR solution as warm start for composite objective

    def neg_composite(w):
        sh, cal, so = composite_score(w, mat)
        return -(w1 * sh + w2 * cal + w3 * so)

    r = minimize(
        neg_composite,
        w0,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    return r.x


def run_aa(prices, strat, folds):
    if isinstance(strat, MultiAssetStrategy):
        return walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
    return walk_forward(prices, lambda _, s=strat: s, folds=folds)


def main():
    print(
        f"H551: MultiObjectiveMDR (composite Sharpe+Calmar+Sortino over H543 55-strategy pool)\n{SEP}"
    )
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
        # H150-H199 (same as H300/H543)
        "H157": lambda tp, tm: VolScaledMomentum(),
        "H175": lambda tp, tm: BettingAgainstBeta(spy_rets=spy_rets, sectors=sectors),
        "H185": lambda tp, tm: WeeklyReversalSectorNeutral(sectors=sectors),
        "H186": lambda tp, tm: ResidualWeeklyReversal(spy_rets=spy_rets, sectors=sectors),
        "H189": lambda tp, tm: ResidualOUMeanReversion(spy_rets=spy_rets, sectors=sectors),
        "H191": lambda tp, tm: PCAResidualReversal(train_prices=tp, sectors=sectors),
        "H192": lambda tp, tm: PCAOUMeanReversion(train_prices=tp, sectors=sectors),
        "H194": lambda tp, tm: LongShortReversal(sectors=sectors),
        "H199": lambda tp, tm: MultiSignalComposite(net_income=ni, sectors=sectors),
        # H251-H298 Set3 best (same as H300/H543)
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
        # NEW: Batch 3 L/S keepers (same as H543)
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

    ret_df = pd.DataFrame(all_rets)
    print(f"\nPool: {len(all_rets)} strategies")
    ew = ret_df.mean(axis=1)
    print(f"Equal-weight: Sharpe {sharpe(ew):.3f}  MaxDD {max_drawdown(ew):.2%}")

    # Grid search over four weight vectors
    grid = [
        (1.0, 0.0, 0.0, "Pure Sharpe"),
        (0.5, 0.5, 0.0, "Sharpe-Calmar"),
        (0.4, 0.4, 0.2, "Primary target"),
        (0.33, 0.33, 0.33, "Equal weights"),
    ]

    print(f"\n{SEP}")
    print("GRID SEARCH: Multi-objective weight vectors")
    print(SEP)

    grid_results = []
    for w1, w2, w3, label in grid:
        print(f"\nSolving ({label}): w1={w1}, w2={w2}, w3={w3} ...")
        w = multi_obj_weights(ret_df, w1=w1, w2=w2, w3=w3)
        port = pd.Series(ret_df.values @ w, index=ret_df.index)
        s_g = sharpe(port)
        c_g = cagr(port)
        dd_g = max_drawdown(port)
        # compute calmar and sortino inline
        cal_cagr_g = c_g
        calmar_g = cal_cagr_g / abs(dd_g) if dd_g < 0 else 0.0
        r_arr = port.values
        ann_mean_g = r_arr.mean() * TRADING_DAYS
        ds_arr = r_arr[r_arr < 0]
        ds_g = np.sqrt((ds_arr**2).mean()) * np.sqrt(TRADING_DAYS) if len(ds_arr) > 0 else 1.0
        sortino_g = ann_mean_g / ds_g if ds_g > 0 else 0.0
        top_w_g = {n: float(v) for n, v in zip(all_rets.keys(), w, strict=True) if v > 0.005}
        print(
            f"  Sharpe {s_g:.3f}  Calmar {calmar_g:.3f}  Sortino {sortino_g:.3f}  "
            f"MaxDD {dd_g:.2%}  CAGR {c_g:.2%}"
        )
        print(f"  Top weights: {top_w_g}")
        grid_results.append(
            {
                "label": label,
                "w1": w1,
                "w2": w2,
                "w3": w3,
                "sharpe": s_g,
                "calmar": calmar_g,
                "sortino": sortino_g,
                "max_dd": dd_g,
                "cagr": c_g,
                "weights": w,
                "top_w": top_w_g,
                "port": port,
            }
        )

    print(f"\n{SEP}")
    print("GRID SUMMARY")
    print(SEP)
    print(
        f"{'Label':<22} {'w1':>5} {'w2':>5} {'w3':>5}  {'Sharpe':>7} {'Calmar':>7} {'Sortino':>8} {'MaxDD':>8}"
    )
    for gr in grid_results:
        print(
            f"  {gr['label']:<20} {gr['w1']:>5.2f} {gr['w2']:>5.2f} {gr['w3']:>5.2f}  "
            f"{gr['sharpe']:>7.3f} {gr['calmar']:>7.3f} {gr['sortino']:>8.3f} {gr['max_dd']:>8.2%}"
        )

    # Select best: Calmar >= 0.90 with highest Sharpe; else highest Calmar
    qualifying = [gr for gr in grid_results if gr["calmar"] >= 0.90]
    if qualifying:
        best = max(qualifying, key=lambda x: x["sharpe"])
        print(f"\nBest qualifying (Calmar>=0.90): {best['label']} (Sharpe {best['sharpe']:.3f})")
    else:
        best = max(grid_results, key=lambda x: x["calmar"])
        print(
            f"\nNo result reached Calmar>=0.90. Using highest-Calmar: {best['label']} (Calmar {best['calmar']:.3f})"
        )

    mdr = best["port"]
    best_w1, best_w2, best_w3 = best["w1"], best["w2"], best["w3"]
    top_w = best["top_w"]

    print(f"\n{SEP}")
    print(f"FINAL RESULT ({best['label']}): w1={best_w1}, w2={best_w2}, w3={best_w3}")
    print(SEP)
    lo, hi = bootstrap_sharpe_ci(mdr, n_boot=5000)
    s = sharpe(mdr)
    c = cagr(mdr)
    dd = max_drawdown(mdr)
    calmar_final = c / abs(dd) if dd < 0 else 0.0
    r_final = mdr.values
    ann_mean_f = r_final.mean() * TRADING_DAYS
    ds_f = r_final[r_final < 0]
    ds_val = np.sqrt((ds_f**2).mean()) * np.sqrt(TRADING_DAYS) if len(ds_f) > 0 else 1.0
    sortino_final = ann_mean_f / ds_val if ds_val > 0 else 0.0
    print(
        f"Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]  CAGR {c:.2%}  "
        f"MaxDD {dd:.2%}  Calmar {calmar_final:.3f}  Sortino {sortino_final:.3f}"
    )
    print(f"Top weights: {top_w}")

    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds_aa):
        mask = (mdr.index >= fold.test_start) & (mdr.index < fold.test_end)
        fr = mdr[mask]
        fs = sharpe(fr) if len(fr) > 10 else 0.0
        fold_sharpes.append(float(fs))
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_aa)
    corr = mdr.corr(wh.combined_returns.reindex(mdr.index))
    print(f"\nH001 correlation: {corr:+.4f}")
    decision = "KEEP" if s > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    # Build grid results string for notes
    grid_results_str = "; ".join(
        f"{gr['label']}(w1={gr['w1']},w2={gr['w2']},w3={gr['w3']}): "
        f"Sh={gr['sharpe']:.3f} Cal={gr['calmar']:.3f} So={gr['sortino']:.3f} DD={gr['max_dd']:.2%}"
        for gr in grid_results
    )

    write_summary(
        hypothesis_id="551",
        name="multi-objective-mdr",
        strategy_class="MultiObjectiveMDR",
        family="Set8 Batch1 Ensemble",
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
            f"Multi-objective MDR (w1={best_w1}, w2={best_w2}, w3={best_w3}): "
            f"composite = w1*Sharpe + w2*Calmar + w3*Sortino. "
            f"Pool: {len(all_rets)} strategies (same as H543). "
            f"Grid results: {grid_results_str}. Top weights: {top_w}."
        ),
        canonical_params={
            "w1": best_w1,
            "w2": best_w2,
            "w3": best_w3,
            "n_strategies": len(all_rets),
            "method": "SLSQP_multi_objective",
            "top_weights": top_w,
        },
    )


if __name__ == "__main__":
    main()
