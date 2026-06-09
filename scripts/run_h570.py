"""H570: LeverageRegimeConditional.

Two-regime leverage using SPY 200d MA as bull/bear classifier:
- Bull (SPY > 200d MA): cap = 2.5x
- Bear (SPY < 200d MA): cap = 1.0x (no leverage)
Vol-targeting still applies within both regimes (6.5% target, 20d lookback).
Same lev-Sharpe SLSQP objective and MDR warm-start as H559.

Run: uv run python scripts/run_h570.py
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
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
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
BORROW_COST_ANNUAL = 0.005

# H559 leverage params
LEV_TARGET_VOL = 0.065
LEV_MAX_FLAT = 2.0  # H559 flat cap
LEV_MAX_BULL = 2.5  # Bull regime cap
LEV_MAX_BEAR = 1.0  # Bear regime cap (no leverage)
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


def apply_vol_leverage_flat(r_arr):
    """Standard H559 vol-targeting leverage (fixed cap 2.0x)."""
    r = pd.Series(r_arr)
    rv = r.rolling(LEV_LOOKBACK, min_periods=LEV_LOOKBACK).std().shift(1) * np.sqrt(TRADING_DAYS)
    lev = (LEV_TARGET_VOL / rv).clip(upper=LEV_MAX_FLAT)
    lev = lev.where(rv.notna() & (rv > 0), 1.0)
    borrow = (lev - 1.0).clip(lower=0.0) * BORROW_COST_ANNUAL / TRADING_DAYS
    return (lev * r - borrow).values


def apply_regime_conditional_leverage(r_arr, spy_close, r_index):
    """Scale leverage cap based on SPY regime (200d MA).

    Bull (SPY > 200d MA): cap = 2.5x
    Bear (SPY <= 200d MA): cap = 1.0x (no leverage)
    """
    r = pd.Series(r_arr, index=r_index)
    spy_ma200 = spy_close.rolling(200, min_periods=100).mean()
    bull_regime = (spy_close > spy_ma200).reindex(r_index).fillna(True)

    rv = r.rolling(LEV_LOOKBACK, min_periods=LEV_LOOKBACK).std().shift(1) * np.sqrt(TRADING_DAYS)
    # Regime-conditional cap
    cap = pd.Series(np.where(bull_regime.values, LEV_MAX_BULL, LEV_MAX_BEAR), index=r_index)
    lev = LEV_TARGET_VOL / rv
    lev = lev.clip(upper=cap)
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


def main():
    print(f"H570: LeverageRegimeConditional\n{SEP}")
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

    # Build SPY close aligned with clean index
    spy_close_aligned = spy["adj_close"].reindex(clean.index).ffill()

    # Compute regime stats
    spy_ma200 = spy_close_aligned.rolling(200, min_periods=100).mean()
    bull_mask = spy_close_aligned > spy_ma200
    bull_frac = bull_mask.mean()
    print(f"\nBull regime fraction (SPY > 200d MA): {bull_frac:.1%}")

    print("\nComputing MDR warm-start...")
    w0 = mdr_warm_start(mat, n)

    # --- Reference: flat cap (H559 style) ---
    def neg_lev_sharpe_flat(w):
        r_base = mat @ w
        r_lev = apply_vol_leverage_flat(r_base)
        ann_mean = r_lev.mean() * TRADING_DAYS
        ann_std = r_lev.std() * np.sqrt(TRADING_DAYS)
        return -(ann_mean / ann_std) if ann_std > 0 else 0.0

    res_flat = minimize(
        neg_lev_sharpe_flat,
        w0,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    w_flat = res_flat.x
    r_flat_base = mat @ w_flat
    r_flat_lev = apply_vol_leverage_flat(r_flat_base)
    sh_flat, cal_flat, cg_flat, dd_flat, _ = portfolio_metrics(r_flat_lev)
    print(
        f"\nFlat cap (H559 ref): Sh_lev={sh_flat:.3f} Cal={cal_flat:.3f} "
        f"CAGR={cg_flat:.2%} DD={dd_flat:.2%}"
    )

    # --- H570 main: regime-conditional leverage ---
    def neg_regime_lev_sharpe(w):
        r_base = mat @ w
        r_lev = apply_regime_conditional_leverage(r_base, spy_close_aligned, clean.index)
        ann_mean = r_lev.mean() * TRADING_DAYS
        ann_std = r_lev.std() * np.sqrt(TRADING_DAYS)
        return -(ann_mean / ann_std) if ann_std > 0 else 0.0

    res_regime = minimize(
        neg_regime_lev_sharpe,
        w0,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    w_regime = res_regime.x
    r_regime_base = mat @ w_regime
    r_regime_lev = apply_regime_conditional_leverage(r_regime_base, spy_close_aligned, clean.index)
    best_port = pd.Series(r_regime_lev, index=clean.index)
    sh_regime, cal_regime, cg_regime, dd_regime, so_regime = portfolio_metrics(r_regime_lev)
    print(
        f"Regime conditional (H570): Sh_lev={sh_regime:.3f} Cal={cal_regime:.3f} "
        f"CAGR={cg_regime:.2%} DD={dd_regime:.2%}"
    )
    print(f"Delta Sh_lev vs flat: {sh_regime - sh_flat:+.3f}")
    print(f"Delta DD_lev vs flat: {(dd_regime - dd_flat) * 100:+.2f}pp")

    print(f"\n{SEP}")
    print("FINAL (regime-conditional leverage)")
    print(SEP)
    lo, hi = bootstrap_sharpe_ci(best_port, n_boot=5000)
    print(
        f"Regime: Sharpe {sh_regime:.3f}  CI [{lo:.3f}, {hi:.3f}]  "
        f"CAGR {cg_regime:.2%}  MaxDD {dd_regime:.2%}  Calmar {cal_regime:.3f}  "
        f"Sortino {so_regime:.3f}"
    )
    top_w = {strategy_names[i]: float(v) for i, v in enumerate(w_regime) if v > 0.005}
    print(f"Top weights: {top_w}")

    print("\nPer-fold Sharpe (regime-conditional):")
    fold_sharpes = []
    for i, fold in enumerate(folds_aa):
        mask = (best_port.index >= fold.test_start) & (best_port.index < fold.test_end)
        fr = best_port[mask]
        fs = sharpe(fr) if len(fr) > 10 else 0.0
        fold_sharpes.append(float(fs))
        # bull fraction in this fold
        bull_frac_fold = bull_mask.reindex(fr.index).fillna(True).mean() if len(fr) > 0 else 0.0
        print(
            f"  Fold {i:>2}: {fs:+.3f}  (bull={bull_frac_fold:.0%})  "
            f"({fold.test_start.date()} -- {fold.test_end.date()})"
        )

    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_aa)
    corr = best_port.corr(wh.combined_returns.reindex(best_port.index))
    print(f"\nH001 correlation: {corr:+.4f}")

    decision = "KEEP" if sh_regime > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="570",
        name="leverage-regime-conditional",
        strategy_class="LeverageRegimeConditional",
        family="Set8 Batch4 Leverage",
        sharpe=sh_regime,
        ci_low=lo,
        ci_high=hi,
        cagr=cg_regime,
        max_drawdown=dd_regime,
        n_trades=0,
        h001_corr=float(corr),
        fold_sharpes=fold_sharpes,
        decision=decision,
        notes=(
            f"Regime-conditional leverage: bull (SPY>200d MA) cap=2.5x, "
            f"bear cap=1.0x. Vol-target={LEV_TARGET_VOL:.1%}/lookback={LEV_LOOKBACK}d. "
            f"Bull regime fraction: {bull_frac:.1%}. "
            f"Flat-cap reference (H559 style): Sh_lev={sh_flat:.3f} Cal={cal_flat:.3f} "
            f"CAGR={cg_flat:.2%} DD={dd_flat:.2%}. "
            f"Regime conditional: Sh_lev={sh_regime:.3f} Cal={cal_regime:.3f} "
            f"CAGR={cg_regime:.2%} DD={dd_regime:.2%}. "
            f"Delta Sh_lev: {sh_regime - sh_flat:+.3f}. "
            f"H559 reference: Sh_lev=1.879. H556 kill bar: Sh_lev=1.790."
        ),
        canonical_params={
            "objective": "lev-sharpe-regime-conditional",
            "leverage_target_vol": LEV_TARGET_VOL,
            "leverage_max_bull": LEV_MAX_BULL,
            "leverage_max_bear": LEV_MAX_BEAR,
            "leverage_lookback": LEV_LOOKBACK,
            "ma_window": 200,
            "bull_regime_fraction": round(float(bull_frac), 4),
            "flat_ref_sh_lev": round(sh_flat, 4),
            "delta_sh_lev_vs_flat": round(sh_regime - sh_flat, 4),
            "top_weights": top_w,
            "n_strategies": n,
        },
    )


if __name__ == "__main__":
    main()
