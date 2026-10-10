"""H568: WalkForwardLeverageParams.

Per-fold leverage parameter selection: sweep 9 (target_vol, cap) combinations on each
training fold, select by training-set lev-Sharpe, then apply to OOS. Portfolio weights
also re-optimized per fold with selected params. Lookback fixed at 20d.
Same 55-strategy pool as H559.

Run: uv run python scripts/run_h568.py
Expected runtime: ~5-6 hours (data load + 9x SLSQP per fold).
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
LEV_LOOKBACK = 20

# Walk-forward param grid: 9 combinations
PARAM_GRID_WF = list(
    itertools.product(
        [0.05, 0.065, 0.08],  # target_vol
        [1.5, 2.0, 3.0],  # cap
    )
)

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


def apply_vol_leverage_p(r_arr, target_vol, cap, lookback):
    """Vol-targeting leverage with explicit params."""
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


def select_and_optimize_fold(train_mat, w0_global, fold_idx):
    """Select best (tv, cap) on training data, re-optimize weights with those params.

    Returns (best_w, best_tv, best_cap, best_train_sh).
    """
    n = train_mat.shape[1]
    best_sh = -np.inf
    best_tv, best_cap = 0.065, 2.0
    best_w = w0_global.copy()

    for tv, cap in PARAM_GRID_WF:

        def neg_lev_sh(w, _tv=tv, _cap=cap):
            r_base = train_mat @ w
            r_lev = apply_vol_leverage_p(r_base, _tv, _cap, LEV_LOOKBACK)
            ann_mean = r_lev.mean() * TRADING_DAYS
            ann_std = r_lev.std() * np.sqrt(TRADING_DAYS)
            return -(ann_mean / ann_std) if ann_std > 0 else 0.0

        res = minimize(
            neg_lev_sh,
            w0_global,
            method="SLSQP",
            bounds=[(0, 1)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
            options={"maxiter": 500, "ftol": 1e-10},
        )
        w_cand = res.x
        r_train_lev = apply_vol_leverage_p(train_mat @ w_cand, tv, cap, LEV_LOOKBACK)
        sh_train = sharpe(pd.Series(r_train_lev))
        if sh_train > best_sh:
            best_sh = sh_train
            best_tv, best_cap = tv, cap
            best_w = w_cand.copy()

    print(
        f"    Fold {fold_idx}: best params tv={best_tv:.3f} cap={best_cap:.1f}x "
        f"train_Sh={best_sh:.3f}"
    )
    return best_w, best_tv, best_cap, best_sh


def main():
    print(f"H568: WalkForwardLeverageParams\n{SEP}")
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

    ret_df = pd.DataFrame(all_rets)
    clean = ret_df.dropna(how="all")
    mat = clean.fillna(0.0).values
    n = mat.shape[1]
    print(f"\nPool: {n} strategies")

    # Global MDR warm-start for cold-start fallback
    print("\nComputing MDR warm-start...")
    w0_global = mdr_warm_start(mat, n)

    # --- Fixed-params reference (H559: 6.5%/2.0x) ---
    lev_target = 0.065
    lev_cap = 2.0

    def neg_lev_sh_fixed(w):
        r_base = mat @ w
        r_lev = apply_vol_leverage_p(r_base, lev_target, lev_cap, LEV_LOOKBACK)
        ann_mean = r_lev.mean() * TRADING_DAYS
        ann_std = r_lev.std() * np.sqrt(TRADING_DAYS)
        return -(ann_mean / ann_std) if ann_std > 0 else 0.0

    res_fixed = minimize(
        neg_lev_sh_fixed,
        w0_global,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    w_fixed = res_fixed.x
    r_fixed_base = mat @ w_fixed
    r_fixed_lev = apply_vol_leverage_p(r_fixed_base, lev_target, lev_cap, LEV_LOOKBACK)
    sh_fx, cal_fx, cg_fx, dd_fx, so_fx = portfolio_metrics(r_fixed_lev)
    print(
        f"\nFixed params (H559 ref): Sh_lev={sh_fx:.3f} Cal={cal_fx:.3f} "
        f"CAGR={cg_fx:.2%} DD={dd_fx:.2%}"
    )

    # --- Walk-forward param selection ---
    print(f"\n{SEP}")
    print(f"WALK-FORWARD PARAM SELECTION ({len(PARAM_GRID_WF)} combos per fold)")
    print(SEP)

    oos_segments = []
    fold_params = []
    fold_sharpes_wf = []

    for i, fold in enumerate(folds_aa):
        print(f"\n  Fold {i} train: {fold.train_start.date()} -- {fold.train_end.date()}")
        print(f"  Fold {i} test:  {fold.test_start.date()} -- {fold.test_end.date()}")

        # Training period matrix (use all available returns in the train window)
        train_mask = (clean.index >= fold.train_start) & (clean.index < fold.train_end)
        train_mat_fold = clean.loc[train_mask].fillna(0.0).values

        if train_mat_fold.shape[0] < 252:
            # Not enough training data — fall back to global params
            best_tv_f, best_cap_f = lev_target, lev_cap
            best_w_f = w0_global.copy()
            print(f"    Fold {i}: insufficient train data, using global params")
        else:
            best_w_f, best_tv_f, best_cap_f, _ = select_and_optimize_fold(
                train_mat_fold, w0_global, i
            )

        fold_params.append((best_tv_f, best_cap_f))

        # Apply to OOS
        oos_mask = (clean.index >= fold.test_start) & (clean.index < fold.test_end)
        oos_mat_fold = clean.loc[oos_mask].fillna(0.0).values
        if oos_mat_fold.shape[0] < 10:
            print(f"    Fold {i}: no OOS data, skipping")
            continue

        r_oos_base = oos_mat_fold @ best_w_f
        r_oos_lev = apply_vol_leverage_p(r_oos_base, best_tv_f, best_cap_f, LEV_LOOKBACK)
        oos_dates = clean.index[oos_mask]
        seg = pd.Series(r_oos_lev, index=oos_dates)
        oos_segments.append(seg)
        fs = sharpe(seg) if len(seg) > 10 else 0.0
        fold_sharpes_wf.append(float(fs))
        print(
            f"    Fold {i} OOS: tv={best_tv_f:.3f} cap={best_cap_f:.1f}x "
            f"Sh_oos={fs:.3f}  n={len(seg)}"
        )

    # Combine OOS segments
    best_port_wf = pd.concat(oos_segments).sort_index()
    sh_wf, cal_wf, cg_wf, dd_wf, so_wf = portfolio_metrics(best_port_wf.values)

    print(f"\n{SEP}")
    print("FINAL (walk-forward params)")
    print(SEP)
    lo, hi = bootstrap_sharpe_ci(best_port_wf, n_boot=5000)
    print(
        f"WF params:   Sharpe {sh_wf:.3f}  CI [{lo:.3f}, {hi:.3f}]  "
        f"CAGR {cg_wf:.2%}  MaxDD {dd_wf:.2%}  Calmar {cal_wf:.3f}  Sortino {so_wf:.3f}"
    )
    print(
        f"Fixed params: Sharpe {sh_fx:.3f}  CAGR {cg_fx:.2%}  MaxDD {dd_fx:.2%}  "
        f"Calmar {cal_fx:.3f}"
    )
    print(f"Delta Sh_lev (WF - fixed): {sh_wf - sh_fx:+.3f}")

    # Per-fold Sharpe
    print("\nPer-fold Sharpe (WF params):")
    for i, fold in enumerate(folds_aa):
        if i < len(fold_sharpes_wf):
            tv_f, cap_f = fold_params[i] if i < len(fold_params) else (0.065, 2.0)
            print(
                f"  Fold {i:>2}: {fold_sharpes_wf[i]:+.3f}  "
                f"(tv={tv_f:.3f} cap={cap_f:.1f}x)  "
                f"({fold.test_start.date()} -- {fold.test_end.date()})"
            )

    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_aa)
    corr = best_port_wf.corr(wh.combined_returns.reindex(best_port_wf.index))
    print(f"\nH001 correlation: {corr:+.4f}")

    decision = "KEEP" if sh_wf > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    params_per_fold_str = "; ".join(
        f"fold{i}: tv={tv:.3f} cap={cap:.1f}x" for i, (tv, cap) in enumerate(fold_params)
    )

    write_summary(
        hypothesis_id="568",
        name="walk-forward-leverage-params",
        strategy_class="WalkForwardLeverageParams",
        family="Set8 Batch4 Leverage",
        sharpe=sh_wf,
        ci_low=lo,
        ci_high=hi,
        cagr=cg_wf,
        max_drawdown=dd_wf,
        n_trades=0,
        h001_corr=float(corr),
        fold_sharpes=fold_sharpes_wf,
        decision=decision,
        notes=(
            f"Walk-forward leverage param selection: 9 combos of tv x cap per fold. "
            f"Fixed-params reference (H559 style): Sh_lev={sh_fx:.3f} Cal={cal_fx:.3f} "
            f"CAGR={cg_fx:.2%} DD={dd_fx:.2%}. "
            f"WF delta vs fixed: {sh_wf - sh_fx:+.3f} Sharpe. "
            f"Lookback fixed at {LEV_LOOKBACK}d. "
            f"Params per fold: {params_per_fold_str}."
        ),
        canonical_params={
            "param_grid": [list(p) for p in PARAM_GRID_WF],
            "lookback": LEV_LOOKBACK,
            "fold_params": [(float(tv), float(cap)) for tv, cap in fold_params],
            "fixed_ref_sh_lev": round(sh_fx, 4),
            "delta_sh_lev": round(sh_wf - sh_fx, 4),
            "n_strategies": n,
        },
    )


if __name__ == "__main__":
    main()
