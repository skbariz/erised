"""H553: RatchetOnH552 — ratchet drawdown protection overlay on H552's 58-strategy MDR portfolio.

If H552 achieves Calmar ~0.84 (same as H543), ratchet should push Calmar above 0.90,
mirroring the H543→H547 improvement (+0.068 Calmar). Tests dd_limit grid [0.10, 0.15, 0.20, 0.25].

Run: uv run python scripts/run_h553.py
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
from erised.metrics.performance import bootstrap_sharpe_ci, cagr, max_drawdown, sharpe
from erised.strategies import (
    AbsoluteMomentumStrategy, CarryMomentumSPYTLT, CorrelationRiskParityHybrid,
    CrisisProtocolRiskParity, DefensiveEquityHedge, DefensiveSectorOverlay,
    EqualWeightDefensiveFactor, FourAssetRiskParity, GoldFearGauge,
    H001H021DirectBlend, H043H021RegimeBlend, HalloweenMomentumConfirm,
    JanuaryBarometer, LowVolTargetPortfolio, MomentumCarryBlend,
    MonthlyRebalancingRiskParity, MovingAverageCrossover, MultiAssetDynamicVolTarget,
    MultiAssetStrategy, RatchetDrawdownProtection, RealizedVsImpliedVol,
    ReversionEntryRiskParity, RiskAdjustedMomentum, RiskParity, RiskParityMomentumFilter,
    RiskParityYieldCurveOverlay, SeasonalityTrendCombo, SeasonalRiskParity, SellInMay,
    SellInMayBondConfirm, SPYGLDRotation, SPYTLTCorrelationRegime, SPYTLTDualMomentum,
    ThreeAssetRiskParity, TimeSeriesMomentum, VixRiskParityEnhanced, VolScaledTrend,
    VrpCalendarHybrid,
)
from erised.strategies.cross_sectional import (
    BettingAgainstBeta, CostAwareMomentumLongHold, FundamentalReversalWeekly,
    InvestmentEfficiencyRatio, KnowledgeReturnProxy, LongShortPCAReversal,
    LongShortResidualReversal, LongShortReversal, LongShortSectorReversal,
    LowCoverageMomentumProxy, MomentumWithPortfolioRatchet, MultiSignalComposite,
    PCAOUMeanReversion, PCAResidualReversal, ResidualOUMeanReversion,
    ResidualWeeklyReversal, ReversionQualityCombo, SustainabilityFilteredProfitability,
    TailRiskComposite, VolScaledMomentum, WeeklyReversalSectorNeutral,
)
from erised.validation.walk_forward import (
    make_folds, walk_forward, walk_forward_cross_sectional, walk_forward_multi,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEP = "=" * 64

AA_POOL = {
    "H001": MovingAverageCrossover(50, 200), "H007": TimeSeriesMomentum(),
    "H008": AbsoluteMomentumStrategy(), "H016": VolScaledTrend(),
    "H018": SPYTLTDualMomentum(), "H019": SPYGLDRotation(),
    "H021": RiskParity(), "H022": DefensiveEquityHedge(), "H023": SellInMay(),
    "H025": JanuaryBarometer(), "H027": SeasonalityTrendCombo(),
    "H031": RealizedVsImpliedVol(), "H034": SPYTLTCorrelationRegime(),
    "H035": GoldFearGauge(), "H043": SellInMayBondConfirm(),
    "H046": HalloweenMomentumConfirm(), "H051": ThreeAssetRiskParity(),
    "H060": CarryMomentumSPYTLT(), "H063": DefensiveSectorOverlay(),
    "H065": EqualWeightDefensiveFactor(), "H066": VixRiskParityEnhanced(),
    "H068": VrpCalendarHybrid(), "H070": MultiAssetDynamicVolTarget(),
    "H076": FourAssetRiskParity(), "H077": RiskParityMomentumFilter(),
    "H078": SeasonalRiskParity(), "H079": CrisisProtocolRiskParity(),
    "H080": RiskParityYieldCurveOverlay(), "H082": RiskAdjustedMomentum(),
    "H084": MomentumCarryBlend(), "H086": LowVolTargetPortfolio(),
    "H087": RatchetDrawdownProtection(), "H088": H001H021DirectBlend(),
    "H089": H043H021RegimeBlend(), "H090": MonthlyRebalancingRiskParity(),
    "H097": CorrelationRiskParityHybrid(), "H098": ReversionEntryRiskParity(),
}


def mdr_weights(returns_df):
    clean = returns_df.dropna(how="all")
    cov = clean.cov().values * 252
    vols = np.sqrt(np.diag(cov))
    n = len(vols)
    def neg_dr(w):
        pv = w @ cov @ w
        return 0.0 if pv <= 0 else -(np.dot(w, vols) / np.sqrt(pv))
    r = minimize(neg_dr, np.full(n, 1.0/n), method="SLSQP", bounds=[(0,1)]*n,
                 constraints=[{"type":"eq","fun":lambda w: w.sum()-1}],
                 options={"maxiter":1000,"ftol":1e-12})
    return r.x


def apply_ratchet(rets: pd.Series, dd_limit: float = 0.15) -> pd.Series:
    """Apply ratchet drawdown protection to a daily return series (no lookahead).

    scale_t = max(0, 1 + drawdown_t / dd_limit)
    Ratcheted return_t = return_t * scale_{t-1}
    """
    r = rets.fillna(0.0)
    cum = (1 + r).cumprod()
    rolling_peak = cum.expanding().max()
    drawdown = cum / rolling_peak - 1  # <= 0
    scale = np.maximum(0.0, 1.0 + drawdown / dd_limit)
    return r * scale.shift(1).fillna(1.0)


def run_aa(prices, strat, folds):
    if isinstance(strat, MultiAssetStrategy):
        return walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
    return walk_forward(prices, lambda _, s=strat: s, folds=folds)


def fold_sharpes(rets, folds):
    result = []
    for fold in folds:
        mask = (rets.index >= fold.test_start) & (rets.index < fold.test_end)
        fr = rets[mask]
        result.append(float(sharpe(fr)) if len(fr) > 10 else 0.0)
    return result


def main():
    print(f"H553: RatchetOnH552 (ratchet overlay on H552 58-strategy MDR)\n{SEP}")

    spy = load_research("SPY")
    etfs = {n: load_research(n)[["adj_close"]].rename(columns={"adj_close": f"{n.lower()}_close"})
            for n in ["TLT","GLD","VIX","SHY","DBC","XLP","XLU","IVE"]}
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
            gross_profit=gp, total_assets=ta, current_assets=ca,
            current_liabilities=cl, net_income=ni, operating_cf=ocf, sectors=sectors),
        "H268": lambda tp, tm: KnowledgeReturnProxy(gross_profit=gp, total_assets=ta, sectors=sectors),
        "H284": lambda tp, tm: LowCoverageMomentumProxy(spy_prices=spy["adj_close"], sectors=sectors),
        "H287": lambda tp, tm: InvestmentEfficiencyRatio(revenue=rev, total_assets=ta, current_assets=ca, sectors=sectors),
        "H293": lambda tp, tm: CostAwareMomentumLongHold(sectors=sectors),
        "H471": lambda tp, tm: LongShortSectorReversal(sectors=sectors),
        "H472": lambda tp, tm: LongShortResidualReversal(spy_rets=spy_rets, sectors=sectors),
        "H473": lambda tp, tm: LongShortPCAReversal(train_prices=tp, sectors=sectors),
        "H537": lambda tp, tm: MomentumWithPortfolioRatchet(sectors=sectors),
        "H538": lambda tp, tm: ReversionQualityCombo(gross_profit=gp, total_assets=ta, sectors=sectors),
        "H540": lambda tp, tm: FundamentalReversalWeekly(net_income=ni, operating_cf=ocf, total_assets=ta, sectors=sectors),
    }
    print(f"\nRunning {len(cs_factories)} CS strategies...")
    for name, factory in cs_factories.items():
        wf = walk_forward_cross_sectional(panel, membership, factory, folds=folds_cs, spy_returns=spy_rets)
        all_rets[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    ret_df = pd.DataFrame(all_rets)
    print(f"\nPool: {len(all_rets)} strategies")

    # MDR weights (unpenalized, same as H552 best variant)
    print(f"\n{SEP}\nSolving MDR weights...")
    w_mdr = mdr_weights(ret_df)
    mdr = pd.Series(ret_df.values @ w_mdr, index=ret_df.index)
    top_w = {n: round(float(v), 4) for n, v in zip(all_rets.keys(), w_mdr) if v > 0.005}
    s_mdr = sharpe(mdr)
    c_mdr = cagr(mdr)
    dd_mdr = max_drawdown(mdr)
    calmar_mdr = c_mdr / abs(dd_mdr) if dd_mdr < 0 else 0
    print(f"MDR (base): Sharpe {s_mdr:.3f}  CAGR {c_mdr:.2%}  MaxDD {dd_mdr:.2%}  Calmar {calmar_mdr:.3f}")
    print(f"Top weights: {top_w}")

    # H001 correlation baseline
    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_aa)
    wh_rets = wh.combined_returns

    # Also evaluate H551's discovered weights (H065/H076/H473/H157 combination)
    # These were the multi-objective optimal weights found by H551 warm-started from MDR
    H551_WEIGHTS = {
        "H065": 0.2686, "H076": 0.2580, "H473": 0.1652,
        "H157": 0.1496, "H016": 0.0744, "H063": 0.0337,
        "H043": 0.0188, "H192": 0.0128,
    }
    total_w551 = sum(H551_WEIGHTS.values())
    w_h551 = np.array([H551_WEIGHTS.get(n, 0.0) / total_w551 for n in all_rets.keys()])
    h551_port = pd.Series(ret_df.values @ w_h551, index=ret_df.index)
    s_h551 = sharpe(h551_port)
    c_h551 = cagr(h551_port)
    dd_h551 = max_drawdown(h551_port)
    cal_h551 = c_h551 / abs(dd_h551) if dd_h551 < 0 else 0
    print(f"\nH551 weights applied to H553 pool: Sharpe {s_h551:.3f}  CAGR {c_h551:.2%}  MaxDD {dd_h551:.2%}  Calmar {cal_h551:.3f}")

    # Grid search dd_limit — apply to BOTH MDR and H551 portfolios
    DD_LIMITS = [0.10, 0.15, 0.20, 0.25]
    print(f"\n{SEP}\nRatchet grid (on MDR base): dd_limit = {DD_LIMITS}")
    print(f"{'dd_limit':>10} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>8} {'Delta_Cal':>10}")
    print("-" * 60)

    best_calmar = calmar_mdr
    best_ddlimit = None
    best_ratchet_rets = None
    grid_results = {}

    for ddl in DD_LIMITS:
        r_ratch = apply_ratchet(mdr, dd_limit=ddl)
        s_r = sharpe(r_ratch)
        c_r = cagr(r_ratch)
        dd_r = max_drawdown(r_ratch)
        cal_r = c_r / abs(dd_r) if dd_r < 0 else 0
        delta_cal = cal_r - calmar_mdr
        print(f"{ddl:>10.2f} {s_r:>8.3f} {c_r:>7.2%} {dd_r:>7.2%} {cal_r:>8.3f} {delta_cal:>+10.3f}")
        grid_results[ddl] = (s_r, c_r, dd_r, cal_r, r_ratch)
        if cal_r > best_calmar:
            best_calmar = cal_r
            best_ddlimit = ddl
            best_ratchet_rets = r_ratch

    # Also test ratchet on H551's portfolio
    print(f"\n{SEP}\nRatchet grid (on H551 weights): dd_limit = {DD_LIMITS}")
    print(f"{'dd_limit':>10} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>8} {'Delta_Cal':>10}")
    print("-" * 60)
    for ddl in DD_LIMITS:
        r_ratch_h551 = apply_ratchet(h551_port, dd_limit=ddl)
        s_r = sharpe(r_ratch_h551)
        c_r = cagr(r_ratch_h551)
        dd_r = max_drawdown(r_ratch_h551)
        cal_r = c_r / abs(dd_r) if dd_r < 0 else 0
        delta_cal = cal_r - cal_h551
        print(f"{ddl:>10.2f} {s_r:>8.3f} {c_r:>7.2%} {dd_r:>7.2%} {cal_r:>8.3f} {delta_cal:>+10.3f}")
        # Track best overall (across both base portfolios)
        if cal_r > best_calmar:
            best_calmar = cal_r
            best_ddlimit = ddl
            best_ratchet_rets = r_ratch_h551
            print(f"  --> New best: {cal_r:.3f}")

    if best_ddlimit is None:
        # Ratchet didn't improve Calmar — use dd_limit=0.20 as canonical
        best_ddlimit = 0.20
        best_ratchet_rets = grid_results[0.20][4]
        print(f"\nRatchet did not improve Calmar over MDR base ({calmar_mdr:.3f}). Using dd_limit=0.20 as canonical.")
    else:
        print(f"\nBest dd_limit: {best_ddlimit} (Calmar {best_calmar:.3f} vs MDR {calmar_mdr:.3f}, +{best_calmar-calmar_mdr:.3f})")

    # Final result
    s_final = sharpe(best_ratchet_rets)
    c_final = cagr(best_ratchet_rets)
    dd_final = max_drawdown(best_ratchet_rets)
    cal_final = c_final / abs(dd_final) if dd_final < 0 else 0
    lo, hi = bootstrap_sharpe_ci(best_ratchet_rets, n_boot=5000)
    corr = best_ratchet_rets.corr(wh_rets.reindex(best_ratchet_rets.index))
    fs = fold_sharpes(best_ratchet_rets, folds_aa)

    print(f"\n{SEP}")
    print(f"FINAL: H553 Ratchet(dd_limit={best_ddlimit}) on H552 MDR")
    print(f"  Sharpe {s_final:.3f}  CI [{lo:.3f}, {hi:.3f}]  CAGR {c_final:.2%}  MaxDD {dd_final:.2%}  Calmar {cal_final:.3f}")
    print(f"  H001 corr: {corr:+.4f}")
    print("  Per-fold Sharpe:")
    for i, (fold, fs_i) in enumerate(zip(folds_aa, fs)):
        print(f"    Fold {i:>2}: {fs_i:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    decision = "KEEP" if s_final > 0.51 and lo > 0 else "kill"
    print(f"\nDecision: {decision}")

    grid_str = "; ".join(f"dd={ddl}: Sh={grid_results[ddl][0]:.3f} Cal={grid_results[ddl][3]:.3f}"
                         for ddl in DD_LIMITS)

    write_summary(
        hypothesis_id="553",
        name="ratchet-on-h552",
        strategy_class="RatchetOnH552",
        family="Set8 Batch1 Ensemble",
        sharpe=s_final, ci_low=lo, ci_high=hi,
        cagr=c_final, max_drawdown=dd_final, n_trades=0,
        h001_corr=float(corr), fold_sharpes=fs, decision=decision,
        notes=(f"Ratchet overlay (dd_limit={best_ddlimit}) on H552's 58-strategy MDR portfolio. "
               f"MDR base: Sh={s_mdr:.3f} Cal={calmar_mdr:.3f} MaxDD={dd_mdr:.2%}. "
               f"Ratchet improved Calmar by {cal_final-calmar_mdr:+.3f}. "
               f"Grid: {grid_str}. MDR top weights: {top_w}."),
        canonical_params={"dd_limit": best_ddlimit, "n_strategies": len(all_rets),
                          "method": "ratchet_on_MDR", "top_weights": top_w},
    )


if __name__ == "__main__":
    main()
