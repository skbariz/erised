"""H569: CostAwareLevSharpe.

Replace flat 0.5% annual borrow cost with historical step-function Fed Funds Rate + 50bps.
Same lev-Sharpe SLSQP objective and MDR warm-start as H559. Robustness test: does H559's
1.879 Sharpe hold under realistic pre-2008 borrow costs (~5-5.5%)?

Run: uv run python scripts/run_h569.py
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
BORROW_COST_ANNUAL_FLAT = 0.005  # H559 flat cost

# H559 leverage params
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


def get_fed_funds_proxy(dates):
    """Step-function approximation of Fed Funds Rate + 50bps spread (annual rate)."""
    rates = pd.Series(np.nan, index=dates)
    # Historical Fed Funds approximations (annual rates)
    periods = [
        ("1993-01-01", "2000-01-01", 0.055),  # 5.5% avg 1993-2000
        ("2000-01-01", "2003-07-01", 0.045),  # cutting post dot-com
        ("2003-07-01", "2004-07-01", 0.010),  # near-zero briefly
        ("2004-07-01", "2007-09-01", 0.050),  # tightening cycle
        ("2007-09-01", "2009-01-01", 0.020),  # cutting for GFC
        ("2009-01-01", "2015-12-01", 0.0025),  # near-zero post-GFC
        ("2015-12-01", "2019-08-01", 0.022),  # tightening 2015-2019
        ("2019-08-01", "2020-03-01", 0.017),  # pre-COVID
        ("2020-03-01", "2022-03-01", 0.001),  # COVID near-zero
        ("2022-03-01", "2024-01-01", 0.040),  # post-COVID tightening
    ]
    for start, end, rate in periods:
        mask = (rates.index >= pd.Timestamp(start)) & (rates.index < pd.Timestamp(end))
        rates[mask] = rate
    rates = rates.fillna(0.025)  # fallback for any remaining gaps
    return rates + 0.005  # add 50bps spread


def apply_vol_leverage_flat(r_arr):
    """Standard H559 vol-targeting leverage with flat 0.5% borrow."""
    r = pd.Series(r_arr)
    rv = r.rolling(LEV_LOOKBACK, min_periods=LEV_LOOKBACK).std().shift(1) * np.sqrt(TRADING_DAYS)
    lev = (LEV_TARGET_VOL / rv).clip(upper=LEV_MAX)
    lev = lev.where(rv.notna() & (rv > 0), 1.0)
    borrow = (lev - 1.0).clip(lower=0.0) * BORROW_COST_ANNUAL_FLAT / TRADING_DAYS
    return (lev * r - borrow).values


def apply_vol_leverage_realistic_cost(r_arr, borrow_rates):
    """Vol-targeting with historical borrow costs (daily series)."""
    r = pd.Series(r_arr)
    rv = r.rolling(LEV_LOOKBACK, min_periods=LEV_LOOKBACK).std().shift(1) * np.sqrt(TRADING_DAYS)
    lev = (LEV_TARGET_VOL / rv).clip(upper=LEV_MAX)
    lev = lev.where(rv.notna() & (rv > 0), 1.0)
    # borrow_rates is a numpy array aligned with r
    borrow = (lev - 1.0).clip(lower=0.0) * borrow_rates / TRADING_DAYS
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
    print(f"H569: CostAwareLevSharpe\n{SEP}")
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

    # Build historical borrow rates series aligned with clean index
    borrow_rates_series = get_fed_funds_proxy(clean.index)
    borrow_rates_arr = borrow_rates_series.values

    # Print borrow cost summary
    print(f"\nBorrow cost summary (mean: {borrow_rates_series.mean():.3%}/yr):")
    for yr in [1995, 2000, 2005, 2010, 2015, 2020, 2022]:
        yr_mask = clean.index.year == yr
        if yr_mask.any():
            avg_rate = borrow_rates_series[yr_mask].mean()
            print(f"  {yr}: {avg_rate:.3%}/yr")

    print("\nComputing MDR warm-start...")
    w0 = mdr_warm_start(mat, n)

    # --- Reference: flat cost (H559 style) ---
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
        f"\nFlat cost (H559 ref): Sh_lev={sh_flat:.3f} Cal={cal_flat:.3f} "
        f"CAGR={cg_flat:.2%} DD={dd_flat:.2%}"
    )

    # --- H569 main: realistic borrow costs ---
    def neg_lev_sharpe_realistic(w):
        r_base = mat @ w
        r_lev = apply_vol_leverage_realistic_cost(r_base, borrow_rates_arr)
        ann_mean = r_lev.mean() * TRADING_DAYS
        ann_std = r_lev.std() * np.sqrt(TRADING_DAYS)
        return -(ann_mean / ann_std) if ann_std > 0 else 0.0

    res_real = minimize(
        neg_lev_sharpe_realistic,
        w0,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    w_real = res_real.x
    r_real_base = mat @ w_real
    r_real_lev = apply_vol_leverage_realistic_cost(r_real_base, borrow_rates_arr)
    best_port = pd.Series(r_real_lev, index=clean.index)
    sh_real, cal_real, cg_real, dd_real, so_real = portfolio_metrics(r_real_lev)
    print(
        f"Realistic cost (H569): Sh_lev={sh_real:.3f} Cal={cal_real:.3f} "
        f"CAGR={cg_real:.2%} DD={dd_real:.2%}"
    )
    print(f"Delta Sh_lev vs flat: {sh_real - sh_flat:+.3f}")

    # Also compute H569 weights applied with flat cost (for comparison)
    r_real_w_flat = apply_vol_leverage_flat(r_real_base)
    sh_xcheck, _, cg_xcheck, dd_xcheck, _ = portfolio_metrics(r_real_w_flat)
    print(
        f"H569 weights + flat cost: Sh_lev={sh_xcheck:.3f} CAGR={cg_xcheck:.2%} DD={dd_xcheck:.2%}"
    )

    print(f"\n{SEP}")
    print("FINAL (realistic costs)")
    print(SEP)
    lo, hi = bootstrap_sharpe_ci(best_port, n_boot=5000)
    print(
        f"Realistic: Sharpe {sh_real:.3f}  CI [{lo:.3f}, {hi:.3f}]  "
        f"CAGR {cg_real:.2%}  MaxDD {dd_real:.2%}  Calmar {cal_real:.3f}  Sortino {so_real:.3f}"
    )
    top_w = {strategy_names[i]: float(v) for i, v in enumerate(w_real) if v > 0.005}
    print(f"Top weights: {top_w}")

    print("\nPer-fold Sharpe (realistic costs):")
    fold_sharpes = []
    for i, fold in enumerate(folds_aa):
        mask = (best_port.index >= fold.test_start) & (best_port.index < fold.test_end)
        fr = best_port[mask]
        fs = sharpe(fr) if len(fr) > 10 else 0.0
        fold_sharpes.append(float(fs))
        # avg borrow in this fold
        avg_borrow = borrow_rates_series[mask].mean() if mask.any() else 0.0
        print(
            f"  Fold {i:>2}: {fs:+.3f}  (avg_borrow={avg_borrow:.2%}/yr)  "
            f"({fold.test_start.date()} -- {fold.test_end.date()})"
        )

    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_aa)
    corr = best_port.corr(wh.combined_returns.reindex(best_port.index))
    print(f"\nH001 correlation: {corr:+.4f}")

    decision = "KEEP" if sh_real > 0.51 and lo > 0 else "kill"
    print(f"Decision: {decision}")

    write_summary(
        hypothesis_id="569",
        name="cost-aware-lev-sharpe",
        strategy_class="CostAwareLevSharpe",
        family="Set8 Batch4 Leverage",
        sharpe=sh_real,
        ci_low=lo,
        ci_high=hi,
        cagr=cg_real,
        max_drawdown=dd_real,
        n_trades=0,
        h001_corr=float(corr),
        fold_sharpes=fold_sharpes,
        decision=decision,
        notes=(
            f"Realistic borrow cost (Fed Funds step-function + 50bps spread). "
            f"Mean borrow rate: {borrow_rates_series.mean():.3%}/yr vs H559 flat 0.500%/yr. "
            f"Flat-cost reference: Sh_lev={sh_flat:.3f} Cal={cal_flat:.3f} "
            f"CAGR={cg_flat:.2%} DD={dd_flat:.2%}. "
            f"Realistic-cost: Sh_lev={sh_real:.3f} Cal={cal_real:.3f} "
            f"CAGR={cg_real:.2%} DD={dd_real:.2%}. "
            f"Delta Sh_lev: {sh_real - sh_flat:+.3f}. "
            f"H559 reference Sh_lev=1.879 (flat cost)."
        ),
        canonical_params={
            "objective": "lev-sharpe-realistic-cost",
            "leverage_target_vol": LEV_TARGET_VOL,
            "leverage_max": LEV_MAX,
            "leverage_lookback": LEV_LOOKBACK,
            "borrow_mean_annual": round(float(borrow_rates_series.mean()), 4),
            "flat_cost_ref_sh_lev": round(sh_flat, 4),
            "delta_sh_lev_vs_flat": round(sh_real - sh_flat, 4),
            "top_weights": top_w,
            "n_strategies": n,
        },
    )


if __name__ == "__main__":
    main()
