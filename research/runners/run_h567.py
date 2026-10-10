"""H567: LevSortinoDownsideVol.

Tests whether replacing standard vol-targeting with downside vol targeting (semi-deviation)
improves Sharpe and/or Calmar vs H559 (lev-Sharpe: 1.879). Combined with a lev-Sortino
SLSQP objective.  Same 55-strategy pool and MDR warm-start as H559.

Run: uv run python scripts/run_h567.py
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
from erised.strategies import (
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

# H559 leverage params (champion)
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
    """MDR solution used as warm-start for SLSQP."""
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
    """Standard H559 vol-targeting leverage for reference."""
    r = pd.Series(r_arr)
    rv = r.rolling(LEV_LOOKBACK, min_periods=LEV_LOOKBACK).std().shift(1) * np.sqrt(TRADING_DAYS)
    lev = (LEV_TARGET_VOL / rv).clip(upper=LEV_MAX)
    lev = lev.where(rv.notna() & (rv > 0), 1.0)
    borrow = (lev - 1.0).clip(lower=0.0) * BORROW_COST_ANNUAL / TRADING_DAYS
    return (lev * r - borrow).values


def apply_downside_vol_leverage(r_arr):
    """Scale leverage by downside vol only — don't de-risk during rallies.

    Semi-deviation (downside std) annualized.
    sqrt(2) adjustment so downside std ≈ total std for normal distributions.
    Fallback to neutral leverage when downside vol is near zero (calm period).
    """
    r = pd.Series(r_arr)
    neg_r = r.where(r < 0, 0.0)
    down_var = (neg_r**2).rolling(LEV_LOOKBACK, min_periods=LEV_LOOKBACK).mean().shift(1)
    down_std = np.sqrt(down_var) * np.sqrt(TRADING_DAYS) * np.sqrt(2)
    # Fallback to neutral leverage when downside vol is near zero (calm period)
    down_std_safe = down_std.where(down_std > 0.005, np.nan)
    lev = (LEV_TARGET_VOL / down_std_safe).clip(upper=LEV_MAX).fillna(1.0)
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


def main():
    print(f"H567: LevSortinoDownsideVol\n{SEP}")
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

    # --- Objective 1: Leveraged Sharpe (H559 reference) ---
    def neg_leveraged_sharpe(w):
        r_base = mat @ w
        r_lev = apply_vol_leverage(r_base)
        ann_mean = r_lev.mean() * TRADING_DAYS
        ann_std = r_lev.std() * np.sqrt(TRADING_DAYS)
        return -(ann_mean / ann_std) if ann_std > 0 else 0.0

    # --- Objective 2: Downside-vol leveraged Sortino (H567 main) ---
    def neg_lev_sortino_downside(w):
        r_base = mat @ w
        r_lev = apply_downside_vol_leverage(r_base)
        ann_mean = r_lev.mean() * TRADING_DAYS
        ds = r_lev[r_lev < 0]
        ds_std = np.sqrt((ds**2).mean()) * np.sqrt(TRADING_DAYS) if len(ds) > 0 else 1.0
        return -(ann_mean / ds_std) if ds_std > 0 else 0.0

    # --- Objective 3: Downside-vol leveraged Sharpe (cross-check) ---
    def neg_lev_sharpe_downside(w):
        r_base = mat @ w
        r_lev = apply_downside_vol_leverage(r_base)
        ann_mean = r_lev.mean() * TRADING_DAYS
        ann_std = r_lev.std() * np.sqrt(TRADING_DAYS)
        return -(ann_mean / ann_std) if ann_std > 0 else 0.0

    def run_optim(objective_fn, label):
        res = minimize(
            objective_fn,
            w0,
            method="SLSQP",
            bounds=[(0, 1)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        w = res.x
        r_base = mat @ w
        sh_b, cal_b, cg_b, dd_b, so_b = portfolio_metrics(r_base)
        r_lev_std = apply_vol_leverage(r_base)
        sh_s, cal_s, cg_s, dd_s, so_s = portfolio_metrics(r_lev_std)
        r_lev_ds = apply_downside_vol_leverage(r_base)
        sh_d, cal_d, cg_d, dd_d, so_d = portfolio_metrics(r_lev_ds)
        # avg leverage comparison
        r = pd.Series(r_base)
        rv_std = r.rolling(LEV_LOOKBACK, min_periods=LEV_LOOKBACK).std().shift(1) * np.sqrt(
            TRADING_DAYS
        )
        neg_r = r.where(r < 0, 0.0)
        dv = (neg_r**2).rolling(LEV_LOOKBACK, min_periods=LEV_LOOKBACK).mean().shift(1)
        ds_std_ser = np.sqrt(dv) * np.sqrt(TRADING_DAYS) * np.sqrt(2)
        ds_std_safe = ds_std_ser.where(ds_std_ser > 0.005, np.nan)
        lev_std = (LEV_TARGET_VOL / rv_std).clip(upper=LEV_MAX).fillna(1.0)
        lev_ds = (LEV_TARGET_VOL / ds_std_safe).clip(upper=LEV_MAX).fillna(1.0)
        avg_lev_std = lev_std.mean()
        avg_lev_ds = lev_ds.mean()
        print(
            f"  [{label}] base: Sh={sh_b:.3f} | "
            f"std_lev: Sh={sh_s:.3f} Cal={cal_s:.3f} CAGR={cg_s:.2%} DD={dd_s:.2%} "
            f"avg_lev={avg_lev_std:.2f}x | "
            f"ds_lev: Sh={sh_d:.3f} Cal={cal_d:.3f} CAGR={cg_d:.2%} DD={dd_d:.2%} "
            f"avg_lev={avg_lev_ds:.2f}x"
        )
        return w, r_base, sh_b, sh_s, cal_s, cg_s, dd_s, so_s, sh_d, cal_d, cg_d, dd_d, so_d

    print(f"\n{SEP}")
    print("OBJECTIVE COMPARISON")
    print(SEP)

    results = []

    w1, r1, sh1_b, sh1_s, cal1_s, cg1_s, dd1_s, so1_s, sh1_d, cal1_d, cg1_d, dd1_d, so1_d = (
        run_optim(neg_leveraged_sharpe, "H559-ref lev-Sharpe (std vol)")
    )
    results.append(
        ("lev-sharpe-std", w1, sh1_b, sh1_s, cal1_s, cg1_s, dd1_s, sh1_d, cal1_d, cg1_d, dd1_d)
    )

    w2, r2, sh2_b, sh2_s, cal2_s, cg2_s, dd2_s, so2_s, sh2_d, cal2_d, cg2_d, dd2_d, so2_d = (
        run_optim(neg_lev_sortino_downside, "H567 lev-Sortino (downside vol)")
    )
    results.append(
        ("lev-sortino-ds", w2, sh2_b, sh2_s, cal2_s, cg2_s, dd2_s, sh2_d, cal2_d, cg2_d, dd2_d)
    )

    w3, r3, sh3_b, sh3_s, cal3_s, cg3_s, dd3_s, so3_s, sh3_d, cal3_d, cg3_d, dd3_d, so3_d = (
        run_optim(neg_lev_sharpe_downside, "H567 lev-Sharpe (downside vol)")
    )
    results.append(
        ("lev-sharpe-ds", w3, sh3_b, sh3_s, cal3_s, cg3_s, dd3_s, sh3_d, cal3_d, cg3_d, dd3_d)
    )

    # Summary table
    print(f"\n{SEP}")
    print("SUMMARY TABLE")
    print(SEP)
    print(
        f"{'Objective':<36} {'Sh_base':>8} {'Sh_std':>8} {'Sh_ds':>8} "
        f"{'Cal_ds':>8} {'CAGR_ds':>9} {'DD_ds':>8}"
    )
    print("-" * 87)
    for label, _w, sh_b, sh_s, _cal_s, _cg_s, _dd_s, sh_d, cal_d, cg_d, dd_d in results:
        print(
            f"  {label:<34} {sh_b:>8.3f} {sh_s:>8.3f} {sh_d:>8.3f} "
            f"{cal_d:>8.3f} {cg_d * 100:>8.2f}%  {dd_d * 100:>7.2f}%"
        )

    # Select best: maximize Sh on downside-vol leverage
    best_idx = max(range(len(results)), key=lambda i: results[i][9])  # sh_d
    best_label, best_w = results[best_idx][0], results[best_idx][1]
    best_r_base = mat @ best_w
    best_r_ds = apply_downside_vol_leverage(best_r_base)
    best_port = pd.Series(best_r_ds, index=clean.index)
    sh_f, cal_f, cg_f, dd_f, so_f = portfolio_metrics(best_r_ds)

    print(f"\nBest (max Sh_ds = {sh_f:.3f}): {best_label}")
    print(f"\n{SEP}")
    print(f"FINAL ({best_label})")
    print(SEP)
    lo, hi = bootstrap_sharpe_ci(best_port, n_boot=5000)
    print(
        f"Downside-lev: Sharpe {sh_f:.3f}  CI [{lo:.3f}, {hi:.3f}]  "
        f"CAGR {cg_f:.2%}  MaxDD {dd_f:.2%}  Calmar {cal_f:.3f}  Sortino {so_f:.3f}"
    )
    top_w = {strategy_names[i]: float(v) for i, v in enumerate(best_w) if v > 0.005}
    print(f"Top weights: {top_w}")

    print("\nPer-fold Sharpe (downside-vol leveraged):")
    fold_sharpes = []
    for i, fold in enumerate(folds_aa):
        mask = (best_port.index >= fold.test_start) & (best_port.index < fold.test_end)
        fr = best_port[mask]
        fs = sharpe(fr) if len(fr) > 10 else 0.0
        fold_sharpes.append(float(fs))
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_aa)
    corr = best_port.corr(wh.combined_returns.reindex(best_port.index))
    print(f"\nH001 correlation: {corr:+.4f}")

    # avg leverage metrics
    r_pd = pd.Series(best_r_base)
    neg_r_pd = r_pd.where(r_pd < 0, 0.0)
    dv2 = (neg_r_pd**2).rolling(LEV_LOOKBACK, min_periods=LEV_LOOKBACK).mean().shift(1)
    ds_std_s = np.sqrt(dv2) * np.sqrt(TRADING_DAYS) * np.sqrt(2)
    ds_std_s2 = ds_std_s.where(ds_std_s > 0.005, np.nan)
    lev_ds_s = (LEV_TARGET_VOL / ds_std_s2).clip(upper=LEV_MAX).fillna(1.0)
    avg_lev = float(lev_ds_s.mean())
    print(f"Avg downside-vol leverage: {avg_lev:.3f}x")

    decision = "KEEP" if sh_f > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    all_obj_str = "; ".join(
        f"{label}: Sh_std={sh_s:.3f}/Sh_ds={sh_d:.3f}/Cal_ds={cal_d:.3f}/CAGR_ds={cg_d:.2%}/DD_ds={dd_d:.2%}"
        for label, _, _, sh_s, cal_s, cg_s, dd_s, sh_d, cal_d, cg_d, dd_d in results
    )

    write_summary(
        hypothesis_id="567",
        name="lev-sortino-downside-vol",
        strategy_class="LevSortinoDownsideVol",
        family="Set8 Batch4 Leverage",
        sharpe=sh_f,
        ci_low=lo,
        ci_high=hi,
        cagr=cg_f,
        max_drawdown=dd_f,
        n_trades=0,
        h001_corr=float(corr),
        fold_sharpes=fold_sharpes,
        decision=decision,
        notes=(
            f"Best objective: {best_label} (selected by max Sh downside-lev). "
            f"Downside vol targeting with lev-Sortino SLSQP. "
            f"Leverage params: {LEV_TARGET_VOL * 100:.1f}%/{LEV_MAX:.1f}x/{LEV_LOOKBACK}d. "
            f"Avg downside-vol leverage: {avg_lev:.3f}x. "
            f"All objectives: {all_obj_str}. "
            f"H559 reference: Sh_lev=1.879 (standard vol-targeting)."
        ),
        canonical_params={
            "objective": best_label,
            "leverage_type": "downside_vol",
            "leverage_target_vol": LEV_TARGET_VOL,
            "leverage_max": LEV_MAX,
            "leverage_lookback": LEV_LOOKBACK,
            "avg_leverage": avg_lev,
            "top_weights": top_w,
            "n_strategies": n,
        },
    )


if __name__ == "__main__":
    main()
