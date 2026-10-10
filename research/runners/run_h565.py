"""H565: Asymmetric Vol Target.

Fast de-lever (5d lookback), slow re-lever (20d lookback).
Uses max(vol_fast, vol_slow) for leverage calculation.
lev-Sharpe objective only. MDR warm-start.

Same 55-strategy pool as H559.

Run: uv run python scripts/run_h565.py
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

# H559 leverage params
LEV_TARGET_VOL = 0.065
LEV_MAX = 2.0
LEV_LOOKBACK_SLOW = 20
LEV_LOOKBACK_FAST = 5

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
    """MDR solution used as warm-start."""
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
    """Standard symmetric vol-targeting leverage (H559 baseline)."""
    r = pd.Series(r_arr)
    rv = r.rolling(LEV_LOOKBACK_SLOW, min_periods=LEV_LOOKBACK_SLOW).std().shift(1) * np.sqrt(
        TRADING_DAYS
    )
    lev = (LEV_TARGET_VOL / rv).clip(upper=LEV_MAX)
    lev = lev.where(rv.notna() & (rv > 0), 1.0)
    borrow = (lev - 1.0).clip(lower=0.0) * BORROW_COST_ANNUAL / TRADING_DAYS
    return (lev * r - borrow).values


def apply_asymmetric_vol_leverage(r_arr):
    """Fast de-lever, slow re-lever.

    Uses max(vol_fast_5d, vol_slow_20d) as the effective vol estimate.
    De-risks quickly when vol spikes; re-levers slowly as vol subsides.
    """
    r = pd.Series(r_arr)
    vol_fast = r.rolling(LEV_LOOKBACK_FAST, min_periods=LEV_LOOKBACK_FAST).std().shift(1) * np.sqrt(
        TRADING_DAYS
    )
    vol_slow = r.rolling(LEV_LOOKBACK_SLOW, min_periods=LEV_LOOKBACK_SLOW).std().shift(1) * np.sqrt(
        TRADING_DAYS
    )
    # Use the higher of the two estimates (conservative)
    vol_eff = vol_fast.where(vol_fast > vol_slow, vol_slow)
    # Fill NaN (first 20 days) with target vol -> neutral leverage of 1.0
    vol_eff = vol_eff.fillna(LEV_TARGET_VOL)
    lev = (LEV_TARGET_VOL / vol_eff).clip(upper=LEV_MAX)
    lev = lev.where(vol_eff > 0, 1.0)
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


def avg_leverage(r_arr):
    """Compute average realized leverage for the asymmetric scheme."""
    r = pd.Series(r_arr)
    vol_fast = r.rolling(LEV_LOOKBACK_FAST, min_periods=LEV_LOOKBACK_FAST).std().shift(1) * np.sqrt(
        TRADING_DAYS
    )
    vol_slow = r.rolling(LEV_LOOKBACK_SLOW, min_periods=LEV_LOOKBACK_SLOW).std().shift(1) * np.sqrt(
        TRADING_DAYS
    )
    vol_eff = vol_fast.where(vol_fast > vol_slow, vol_slow)
    vol_eff = vol_eff.fillna(LEV_TARGET_VOL)
    lev = (LEV_TARGET_VOL / vol_eff).clip(upper=LEV_MAX)
    lev = lev.where(vol_eff > 0, 1.0)
    return float(lev.mean())


def optimize(mat, w0, label):
    """Optimize lev-Sharpe with asymmetric leverage."""
    n = mat.shape[1]

    def neg_leveraged_sharpe(w):
        r_base = mat @ w
        r_lev = apply_asymmetric_vol_leverage(r_base)
        ann_mean = r_lev.mean() * TRADING_DAYS
        ann_std = r_lev.std() * np.sqrt(TRADING_DAYS)
        return -(ann_mean / ann_std) if ann_std > 0 else 0.0

    res = minimize(
        neg_leveraged_sharpe,
        w0,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    w = res.x
    r = mat @ w
    sh, cal, cg, dd, so = portfolio_metrics(r)
    r_lev = apply_asymmetric_vol_leverage(r)
    sh_lev, cal_lev, cg_lev, dd_lev, so_lev = portfolio_metrics(r_lev)
    avg_lev = avg_leverage(r)
    print(
        f"  [{label}] base: Sh={sh:.3f} DD={dd:.2%} | "
        f"lev: Sh={sh_lev:.3f} Cal={cal_lev:.3f} CAGR={cg_lev:.2%} DD={dd_lev:.2%} avg_lev={avg_lev:.3f}x"
    )
    return w, r, sh, cal, cg, dd, so, sh_lev, cal_lev, cg_lev, dd_lev, so_lev, avg_lev


def main():
    print(f"H565: Asymmetric Vol Target\n{SEP}")
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

    # Also compute H559-style symmetric baseline for comparison
    print(f"\n{SEP}")
    print("COMPARISON: Symmetric vs Asymmetric Vol Targeting")
    print(SEP)

    # Symmetric baseline (H559 lev-Sharpe approach)
    def neg_sym_sharpe(w):
        r_base = mat @ w
        r_lev = apply_vol_leverage(r_base)
        ann_mean = r_lev.mean() * TRADING_DAYS
        ann_std = r_lev.std() * np.sqrt(TRADING_DAYS)
        return -(ann_mean / ann_std) if ann_std > 0 else 0.0

    n_strats = mat.shape[1]
    res_sym = minimize(
        neg_sym_sharpe,
        w0,
        method="SLSQP",
        bounds=[(0, 1)] * n_strats,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    w_sym = res_sym.x
    r_sym = mat @ w_sym
    sh_sym, cal_sym, cg_sym, dd_sym, so_sym = portfolio_metrics(r_sym)
    r_sym_lev = apply_vol_leverage(r_sym)
    sh_sym_lev, cal_sym_lev, cg_sym_lev, dd_sym_lev, _ = portfolio_metrics(r_sym_lev)
    r_sym_arr = pd.Series(r_sym)
    rv_sym = r_sym_arr.rolling(LEV_LOOKBACK_SLOW, min_periods=LEV_LOOKBACK_SLOW).std().shift(
        1
    ) * np.sqrt(TRADING_DAYS)
    lev_sym = (LEV_TARGET_VOL / rv_sym).clip(upper=LEV_MAX)
    lev_sym = lev_sym.where(rv_sym.notna() & (rv_sym > 0), 1.0)
    avg_lev_sym = float(lev_sym.mean())
    print(
        f"  [symmetric-20d] base: Sh={sh_sym:.3f} | "
        f"lev: Sh={sh_sym_lev:.3f} Cal={cal_sym_lev:.3f} CAGR={cg_sym_lev:.2%} "
        f"DD={dd_sym_lev:.2%} avg_lev={avg_lev_sym:.3f}x"
    )

    # Asymmetric result
    (
        w_asym,
        r_asym,
        sh_asym,
        cal_asym,
        cg_asym,
        dd_asym,
        so_asym,
        sh_asym_lev,
        cal_asym_lev,
        cg_asym_lev,
        dd_asym_lev,
        so_asym_lev,
        avg_lev_asym,
    ) = optimize(mat, w0, "asymmetric (fast=5d/slow=20d)")

    print(f"\n{SEP}")
    print("FINAL (asymmetric vol target)")
    print(SEP)

    best_port_lev = pd.Series(apply_asymmetric_vol_leverage(r_asym), index=clean.index)
    lo, hi = bootstrap_sharpe_ci(best_port_lev, n_boot=5000)
    print(
        f"Asymmetric: Sharpe {sh_asym_lev:.3f}  CI [{lo:.3f}, {hi:.3f}]  "
        f"CAGR {cg_asym_lev:.2%}  MaxDD {dd_asym_lev:.2%}  "
        f"Calmar {cal_asym_lev:.3f}  avg_lev {avg_lev_asym:.3f}x"
    )
    print(
        f"Symmetric baseline: Sharpe {sh_sym_lev:.3f}  CAGR {cg_sym_lev:.2%}  "
        f"MaxDD {dd_sym_lev:.2%}  avg_lev {avg_lev_sym:.3f}x"
    )
    top_w = {strategy_names[i]: float(v) for i, v in enumerate(w_asym) if v > 0.005}
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

    decision = "KEEP" if sh_asym_lev > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="565",
        name="asymmetric-vol-target",
        strategy_class="AsymmetricVolTarget",
        family="Set8 Batch4 Ensemble",
        sharpe=sh_asym_lev,
        ci_low=lo,
        ci_high=hi,
        cagr=cg_asym_lev,
        max_drawdown=dd_asym_lev,
        n_trades=0,
        h001_corr=float(corr),
        fold_sharpes=fold_sharpes,
        decision=decision,
        notes=(
            f"Asymmetric vol target: fast={LEV_LOOKBACK_FAST}d / slow={LEV_LOOKBACK_SLOW}d. "
            f"avg_lev_asym={avg_lev_asym:.3f}x vs avg_lev_sym={avg_lev_sym:.3f}x. "
            f"Symmetric baseline: Sh_lev={sh_sym_lev:.3f}/DD_lev={dd_sym_lev:.2%}. "
            f"H559 reference Sh_lev=1.879/DD_lev=-9.73%."
        ),
        canonical_params={
            "objective": "lev_sharpe_asymmetric",
            "vol_lookback_fast": LEV_LOOKBACK_FAST,
            "vol_lookback_slow": LEV_LOOKBACK_SLOW,
            "leverage_target_vol": LEV_TARGET_VOL,
            "leverage_max": LEV_MAX,
            "avg_lev": avg_lev_asym,
            "top_weights": top_w,
            "n_strategies": n,
        },
    )


if __name__ == "__main__":
    main()
