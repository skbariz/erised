"""H579: ExpandedPoolLevSharpe -- H561's optimal params on 63-strategy pool.

Extends H559's 55-strategy pool with 8 new strategies from H571-H578:
AA additions: QQQTLTRiskParity, IWMTLTRiskParity, QQQIWMTLTRiskParity,
              VNQTLTRiskParity, SectorMomentumVolTarget
CS additions: QuarterlyNIMomentum, ThreeFactorDailyReversal,
              QualityMomentumWeeklyRotation

Uses H561's optimal leverage params: target_vol=5%, cap=3.0x, lookback=20d.
The optimizer will zero-weight strategies that don't add portfolio value.

Run: uv run python scripts/run_h579.py
Expected runtime: ~4-5 hours.
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
    IWMTLTRiskParity,
    JanuaryBarometer,
    LowVolTargetPortfolio,
    MomentumCarryBlend,
    MonthlyRebalancingRiskParity,
    MovingAverageCrossover,
    MultiAssetDynamicVolTarget,
    MultiAssetStrategy,
    QQQIWMTLTRiskParity,
    QQQTLTRiskParity,
    RatchetDrawdownProtection,
    RealizedVsImpliedVol,
    ReversionEntryRiskParity,
    RiskAdjustedMomentum,
    RiskParity,
    RiskParityMomentumFilter,
    RiskParityYieldCurveOverlay,
    SeasonalityTrendCombo,
    SeasonalRiskParity,
    SectorMomentumVolTarget,
    SellInMay,
    SellInMayBondConfirm,
    SPYGLDRotation,
    SPYTLTCorrelationRegime,
    SPYTLTDualMomentum,
    ThreeAssetRiskParity,
    TimeSeriesMomentum,
    VixRiskParityEnhanced,
    VNQTLTRiskParity,
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
    QualityMomentumWeeklyRotation,
    QuarterlyNIMomentum,
    ResidualOUMeanReversion,
    ResidualWeeklyReversal,
    SustainabilityFilteredProfitability,
    TailRiskComposite,
    ThreeFactorDailyReversal,
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

# H561 optimal leverage params
LEV_TARGET_VOL = 0.050
LEV_MAX = 3.0
LEV_LOOKBACK = 20

AA_ETF_NAMES = ["TLT", "GLD", "VIX", "SHY", "DBC", "XLP", "XLU", "IVE", "IVW", "QQQ", "IWM", "VNQ"]


def run_aa(prices, strat, folds):
    if isinstance(strat, MultiAssetStrategy):
        return walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
    return walk_forward(prices, lambda _, s=strat: s, folds=folds)


def mdr_warm_start(mat, n):
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
    r = pd.Series(r_arr)
    rv = r.rolling(LEV_LOOKBACK, min_periods=LEV_LOOKBACK).std().shift(1) * np.sqrt(TRADING_DAYS)
    lev = (LEV_TARGET_VOL / rv).clip(upper=LEV_MAX)
    lev = lev.where(rv.notna() & (rv > 0), 1.0)
    borrow = (lev - 1.0).clip(lower=0.0) * BORROW_COST_ANNUAL / TRADING_DAYS
    return (lev * r - borrow).values


def portfolio_metrics(r_arr):
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
    print(f"H579: ExpandedPoolLevSharpe (63-strategy pool, tv={LEV_TARGET_VOL*100:.0f}%/cap={LEV_MAX:.1f}x/lb={LEV_LOOKBACK}d)\n{SEP}")
    spy = load_research("SPY")
    etfs = {
        n: load_research(n)[["adj_close"]].rename(columns={"adj_close": f"{n.lower()}_close"})
        for n in AA_ETF_NAMES
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

    # Factor returns for H575
    iwm_rets = etfs["IWM"]["iwm_close"].pct_change().fillna(0.0)
    ive_rets = etfs["IVE"]["ive_close"].pct_change().fillna(0.0)
    ivw_rets = etfs["IVW"]["ivw_close"].pct_change().fillna(0.0)
    size_rets = (iwm_rets - spy_rets).fillna(0.0)
    value_rets = (ive_rets - ivw_rets).fillna(0.0)

    print(f"Panel: {panel.shape[0]} days, {panel.shape[1]} tickers")

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
        # New AA strategies (H571-H578)
        "H571": QQQTLTRiskParity(),
        "H572": IWMTLTRiskParity(),
        "H573": QQQIWMTLTRiskParity(),
        "H577": SectorMomentumVolTarget(),
        "H578": VNQTLTRiskParity(),
    }

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
            gross_profit=gp, total_assets=ta, current_assets=ca,
            current_liabilities=cl, net_income=ni, operating_cf=ocf, sectors=sectors,
        ),
        "H268": lambda tp, tm: KnowledgeReturnProxy(gross_profit=gp, total_assets=ta, sectors=sectors),
        "H284": lambda tp, tm: LowCoverageMomentumProxy(spy_prices=spy["adj_close"], sectors=sectors),
        "H287": lambda tp, tm: InvestmentEfficiencyRatio(
            revenue=rev, total_assets=ta, current_assets=ca, sectors=sectors
        ),
        "H293": lambda tp, tm: CostAwareMomentumLongHold(sectors=sectors),
        "H471": lambda tp, tm: LongShortSectorReversal(sectors=sectors),
        "H472": lambda tp, tm: LongShortResidualReversal(spy_rets=spy_rets, sectors=sectors),
        "H473": lambda tp, tm: LongShortPCAReversal(train_prices=tp, sectors=sectors),
        # New CS strategies (H574-H576)
        "H574": lambda tp, tm: QuarterlyNIMomentum(net_income=ni, total_assets=ta, sectors=sectors),
        "H575": lambda tp, tm: ThreeFactorDailyReversal(
            spy_rets=spy_rets, size_rets=size_rets, value_rets=value_rets, sectors=sectors
        ),
        "H576": lambda tp, tm: QualityMomentumWeeklyRotation(
            gross_profit=gp, total_assets=ta, sectors=sectors
        ),
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
    mat = clean.fillna(0.0).to_numpy(dtype="float64")
    n = mat.shape[1]
    print(f"\nPool: {n} strategies")

    print("\nComputing MDR warm-start...")
    w0 = mdr_warm_start(mat, n)

    print(f"\n{SEP}")
    print(f"OPTIMIZING: Leveraged Sharpe ({LEV_TARGET_VOL*100:.0f}%/{LEV_MAX:.1f}x/{LEV_LOOKBACK}d)")
    print(SEP)

    def neg_leveraged_sharpe(w):
        r_base = mat @ w
        r_lev = apply_vol_leverage(r_base)
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
    best_w = res.x
    best_r = mat @ best_w
    sh_base, cal_base, cg_base, dd_base, so_base = portfolio_metrics(best_r)
    r_lev = apply_vol_leverage(best_r)
    sh_lev, cal_lev, cg_lev, dd_lev, so_lev = portfolio_metrics(r_lev)

    print(f"  base: Sh={sh_base:.3f} Cal={cal_base:.3f} CAGR={cg_base:.2%} DD={dd_base:.2%}")
    print(f"  lev:  Sh={sh_lev:.3f} Cal={cal_lev:.3f} CAGR={cg_lev:.2%} DD={dd_lev:.2%}")

    best_port_lev = pd.Series(r_lev, index=clean.index)
    lo, hi = bootstrap_sharpe_ci(best_port_lev, n_boot=5000)
    print(f"\nLeveraged: Sharpe {sh_lev:.3f}  CI [{lo:.3f}, {hi:.3f}]  CAGR {cg_lev:.2%}  MaxDD {dd_lev:.2%}  Calmar {cal_lev:.3f}")

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

    decision = "KEEP" if sh_lev > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    # Note new strategies and their weights
    new_strats = {k: top_w.get(k, 0.0) for k in ["H571", "H572", "H573", "H574", "H575", "H576", "H577", "H578"]}
    print(f"\nNew strategy weights: {new_strats}")

    write_summary(
        hypothesis_id="579",
        name="expanded-pool-lev-sharpe",
        strategy_class="ExpandedPoolLevSharpe",
        family="Set8 Batch2 Ensemble",
        sharpe=sh_lev,
        ci_low=lo,
        ci_high=hi,
        cagr=cg_lev,
        max_drawdown=dd_lev,
        n_trades=0,
        h001_corr=float(corr),
        fold_sharpes=fold_sharpes,
        decision=decision,
        notes=(
            f"63-strategy pool (55 original + 8 new: {list(new_strats.keys())}). "
            f"H561 optimal params: {LEV_TARGET_VOL*100:.0f}%/{LEV_MAX:.1f}x/{LEV_LOOKBACK}d. "
            f"Base metrics: Sh={sh_base:.3f} Cal={cal_base:.3f} CAGR={cg_base:.2%} DD={dd_base:.2%}. "
            f"New strategy weights: {new_strats}. "
            f"H561 reference Sh_lev=1.921."
        ),
        canonical_params={
            "leverage_target_vol": LEV_TARGET_VOL,
            "leverage_max": LEV_MAX,
            "leverage_lookback": LEV_LOOKBACK,
            "top_weights": top_w,
            "n_strategies": n,
            "new_strategy_weights": new_strats,
        },
    )


if __name__ == "__main__":
    main()
