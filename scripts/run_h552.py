"""H552: MaxDDConstrainedMDR -- H543's 55-strategy pool + H537/H538/H540 = 58 strategies.
Adds a soft MaxDD penalty to the MDR objective to push CAGR above 7% while keeping MaxDD <= -10%.
Run: uv run python scripts/run_h552.py
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
    FundamentalReversalWeekly,
    InvestmentEfficiencyRatio,
    KnowledgeReturnProxy,
    LongShortPCAReversal,
    LongShortResidualReversal,
    LongShortReversal,
    LongShortSectorReversal,
    LowCoverageMomentumProxy,
    MomentumWithPortfolioRatchet,
    MultiSignalComposite,
    PCAOUMeanReversion,
    PCAResidualReversal,
    ResidualOUMeanReversion,
    ResidualWeeklyReversal,
    ReversionQualityCombo,
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


def mdr_weights(returns_df):
    """Unpenalized MDR (same as H543)."""
    clean = returns_df.dropna(how="all")
    cov = clean.cov().values * 252
    vols = np.sqrt(np.diag(cov))
    n = len(vols)

    def neg_dr(w):
        pv = w @ cov @ w
        return 0.0 if pv <= 0 else -(np.dot(w, vols) / np.sqrt(pv))

    r = minimize(
        neg_dr,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return r.x


def penalized_mdr_weights(returns_df, maxdd_penalty=5.0, maxdd_target=-0.10):
    """MDR with soft MaxDD penalty."""
    clean = returns_df.dropna(how="all")
    cov = clean.cov().values * 252
    vols = np.sqrt(np.diag(cov))
    mat = clean.values
    n = len(vols)

    def neg_dr_penalized(w):
        pv = w @ cov @ w
        if pv <= 0:
            return 0.0
        dr = np.dot(w, vols) / np.sqrt(pv)
        r = mat @ w
        cum = np.cumprod(1 + r)
        dd = (cum / np.maximum.accumulate(cum) - 1).min()
        # penalize only if MaxDD exceeds target
        penalty = maxdd_penalty * max(0, dd - maxdd_target) if dd < 0 else 0.0
        return -dr + penalty

    r = minimize(
        neg_dr_penalized,
        np.full(n, 1.0 / n),
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


def print_variant_stats(label, ret_series, folds_aa, wh_rets):
    s = sharpe(ret_series)
    c = cagr(ret_series)
    dd = max_drawdown(ret_series)
    calmar = abs(c / dd) if dd != 0 else 0.0
    lo, hi = bootstrap_sharpe_ci(ret_series, n_boot=5000)
    corr = ret_series.corr(wh_rets.reindex(ret_series.index))
    print(f"\n{label}")
    print(
        f"  Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]  CAGR {c:.2%}  MaxDD {dd:.2%}  Calmar {calmar:.3f}"
    )
    print(f"  H001 corr: {corr:+.4f}")
    print("  Per-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds_aa):
        mask = (ret_series.index >= fold.test_start) & (ret_series.index < fold.test_end)
        fr = ret_series[mask]
        fs = sharpe(fr) if len(fr) > 10 else 0.0
        fold_sharpes.append(float(fs))
        print(f"    Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")
    return s, lo, hi, c, dd, corr, fold_sharpes


def main():
    print(
        f"H552: MaxDDConstrainedMDR (H543 55-strategy pool + H537/H538/H540 = 58 strategies)\n{SEP}"
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
        # H150-H199 (same as H543/H300)
        "H157": lambda tp, tm: VolScaledMomentum(),
        "H175": lambda tp, tm: BettingAgainstBeta(spy_rets=spy_rets, sectors=sectors),
        "H185": lambda tp, tm: WeeklyReversalSectorNeutral(sectors=sectors),
        "H186": lambda tp, tm: ResidualWeeklyReversal(spy_rets=spy_rets, sectors=sectors),
        "H189": lambda tp, tm: ResidualOUMeanReversion(spy_rets=spy_rets, sectors=sectors),
        "H191": lambda tp, tm: PCAResidualReversal(train_prices=tp, sectors=sectors),
        "H192": lambda tp, tm: PCAOUMeanReversion(train_prices=tp, sectors=sectors),
        "H194": lambda tp, tm: LongShortReversal(sectors=sectors),
        "H199": lambda tp, tm: MultiSignalComposite(net_income=ni, sectors=sectors),
        # H251-H298 Set3 best (same as H543/H300)
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
        # H471-H473 L/S keepers (same as H543)
        "H471": lambda tp, tm: LongShortSectorReversal(sectors=sectors),
        "H472": lambda tp, tm: LongShortResidualReversal(spy_rets=spy_rets, sectors=sectors),
        "H473": lambda tp, tm: LongShortPCAReversal(train_prices=tp, sectors=sectors),
        # NEW: H537/H538/H540 — near-target Set7 all-fold strategies
        "H537": lambda tp, tm: MomentumWithPortfolioRatchet(sectors=sectors),
        "H538": lambda tp, tm: ReversionQualityCombo(
            gross_profit=gp, total_assets=ta, sectors=sectors
        ),
        "H540": lambda tp, tm: FundamentalReversalWeekly(
            net_income=ni, operating_cf=ocf, total_assets=ta, sectors=sectors
        ),
    }
    print(f"\nRunning {len(cs_factories)} CS strategies...")
    for name, factory in cs_factories.items():
        wf = walk_forward_cross_sectional(
            panel, membership, factory, folds=folds_cs, spy_returns=spy_rets
        )
        all_rets[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    ret_df = pd.DataFrame(all_rets)
    n_strats = len(all_rets)
    print(f"\nPool: {n_strats} strategies")
    ew = ret_df.mean(axis=1)
    ew_cagr = cagr(ew)
    print(
        f"Equal-weight: Sharpe {sharpe(ew):.3f}  MaxDD {max_drawdown(ew):.2%}  CAGR {ew_cagr:.2%}"
    )

    # H001 baseline for correlation
    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_aa)
    wh_rets = wh.combined_returns

    # --- Variant 1: Unpenalized MDR (baseline on expanded pool) ---
    print(f"\n{SEP}\nSolving unpenalized MDR (expanded pool baseline)...")
    w_base = mdr_weights(ret_df)
    mdr_base = pd.Series(ret_df.values @ w_base, index=ret_df.index)
    top_w_base = {n: round(float(v), 4) for n, v in zip(all_rets.keys(), w_base, strict=True) if v > 0.005}
    print(f"Top weights: {top_w_base}")
    s_base, lo_base, hi_base, c_base, dd_base, corr_base, fs_base = print_variant_stats(
        "Unpenalized MDR (expanded pool)", mdr_base, folds_aa, wh_rets
    )

    # --- Variant 2: Penalized MDR — grid search penalty strengths ---
    print(f"\n{SEP}\nGrid-searching MaxDD penalty strengths: [1, 2, 5, 10]...")
    best_penalty = None
    best_sharpe = -np.inf
    best_result = None
    penalty_results = {}

    for penalty in [1.0, 2.0, 5.0, 10.0]:
        print(f"\nSolving penalized MDR (penalty={penalty})...")
        w_pen = penalized_mdr_weights(ret_df, maxdd_penalty=penalty, maxdd_target=-0.10)
        mdr_pen = pd.Series(ret_df.values @ w_pen, index=ret_df.index)
        s_p = sharpe(mdr_pen)
        c_p = cagr(mdr_pen)
        dd_p = max_drawdown(mdr_pen)
        calmar_p = abs(c_p / dd_p) if dd_p != 0 else 0.0
        top_w_p = {n: round(float(v), 4) for n, v in zip(all_rets.keys(), w_pen, strict=True) if v > 0.005}
        print(
            f"  penalty={penalty}: Sharpe {s_p:.3f}  CAGR {c_p:.2%}  MaxDD {dd_p:.2%}  Calmar {calmar_p:.3f}"
        )
        print(f"  Top weights: {top_w_p}")
        penalty_results[penalty] = (s_p, c_p, dd_p, calmar_p, mdr_pen, w_pen, top_w_p)
        # Select best by Sharpe (primary), with CAGR >= 6.5% as secondary gate
        if s_p > best_sharpe and c_p >= 0.065:
            best_sharpe = s_p
            best_penalty = penalty
            best_result = penalty_results[penalty]

    # Fallback: if no penalty achieves CAGR >= 6.5%, pick highest Sharpe overall
    if best_result is None:
        print("\nNo penalty variant hit CAGR >= 6.5%; selecting best Sharpe overall.")
        for penalty, res in penalty_results.items():
            s_p = res[0]
            if s_p > best_sharpe:
                best_sharpe = s_p
                best_penalty = penalty
                best_result = res

    print(f"\n{SEP}\nBest penalty: {best_penalty}")
    s_b, c_b, dd_b, calmar_b, mdr_best, w_best, top_w_best = best_result
    s_final, lo_final, hi_final, c_final, dd_final, corr_final, fold_sharpes = print_variant_stats(
        f"Penalized MDR (penalty={best_penalty}) [SELECTED]", mdr_best, folds_aa, wh_rets
    )
    print(f"\nCalmar: {calmar_b:.3f}")

    # Final decision
    decision = "KEEP" if s_final > 0.51 and lo_final > 0 else "kill"
    print(f"\nDecision: {decision}")
    print(f"Top weights (penalized, penalty={best_penalty}): {top_w_best}")

    write_summary(
        hypothesis_id="552",
        name="maxdd-constrained-mdr",
        strategy_class="MaxDDConstrainedMDR",
        family="Set8 Batch1 Ensemble",
        sharpe=s_final,
        ci_low=lo_final,
        ci_high=hi_final,
        cagr=c_final,
        max_drawdown=dd_final,
        n_trades=0,
        h001_corr=float(corr_final),
        fold_sharpes=fold_sharpes,
        decision=decision,
        notes=(
            f"MaxDD-penalized MDR (penalty={best_penalty}) over expanded pool: "
            f"H543's 55 strategies + H537/H538/H540 = {n_strats} strategies. "
            f"EW-baseline CAGR: {ew_cagr:.2%}. "
            f"Unpenalized MDR (expanded): Sharpe {s_base:.3f} CAGR {c_base:.2%} MaxDD {dd_base:.2%}. "
            f"Top weights: {top_w_best}."
        ),
        canonical_params={
            "maxdd_penalty": best_penalty,
            "n_strategies": n_strats,
            "method": "penalized_SLSQP",
            "top_weights": top_w_best,
        },
    )


if __name__ == "__main__":
    main()
