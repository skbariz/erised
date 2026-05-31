"""H449: UltimateSet6MDR -- MDR over all kept H401-H448 + H300/H350 ETF pool.

Grand MDR incorporating Set 6 CS strategies as new diversifiers.
Tests if Set 6 near-zero-H001-corr signals beat H300/H350 (Sharpe 1.525).

Run: uv run python scripts/run_h449.py
Expected runtime: 3-4 hours (runs all Set 6 CS components).
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
    MovingAverageCrossover, RiskParity, SellInMayBondConfirm, SeasonalRiskParity,
    HalloweenMomentumConfirm, MultiAssetDynamicVolTarget, LowVolTargetPortfolio,
    RatchetDrawdownProtection, CorrelationRiskParityHybrid, CarryMomentumSPYTLT,
    CrisisProtocolRiskParity, GoldFearGauge, MultiAssetStrategy,
    FundingStressMonitor, CreditRegimeRiskParity, VRPRiskParityScaled,
)
from quant.strategies.cross_sectional import (
    # Batch 1 (earnings quality)
    AccrualQualityPort, EarningsMomentumQuality, TorpedoRiskAvoid,
    RevenueEarningsAcceleration, CashEarningsPersistence,
    # Batch 2 (disclosure quality)
    EarningsPersistencePort, CashFlowReliabilityPort, BookValueGrowthPort,
    BalanceSheetStrengthPort, WorkingCapitalHealthPort,
    # Batch 3 (sentiment proxies -- includes stars H412/H415)
    ConsistentGrowthStreak, FundamentalDivergenceAvoid, MarginExpansionPort,
    EarningsVariabilityAvoid, StableMarginMomentum,
    # Batch 4 (product-market)
    DifferentiatedMarginPort, CompetitivePressureAvoid, PeerGroupMomentum,
    CashDefenseProxy, MarketShareProxy,
    # Batch 6 (innovation -- includes star H429)
    AssetEfficiencyGrowth, CapitalProductivityPort, GrowthReinvestmentQuality,
    SectorRelativeMarginProxy, OperatingLeverageImprovement,
)
from quant.validation.walk_forward import (
    make_folds, walk_forward, walk_forward_cross_sectional, walk_forward_multi,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEP = "=" * 64


def max_diversification_weights(returns_df: pd.DataFrame) -> np.ndarray:
    clean = returns_df.dropna(how="all")
    cov = clean.cov().values * 252
    vols = np.sqrt(np.diag(cov))
    n = len(vols)

    def neg_dr(w: np.ndarray) -> float:
        port_var = w @ cov @ w
        if port_var <= 0:
            return 0.0
        return -(np.dot(w, vols) / np.sqrt(port_var))

    result = minimize(
        neg_dr, np.full(n, 1.0 / n), method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return result.x


def main():
    print(f"H449: UltimateSet6MDR (grand MDR over H300/H350 pool + Set 6 CS)\n{SEP}")

    # --- Load ETF data ---
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    gld = load_research("GLD")[["adj_close"]].rename(columns={"adj_close": "gld_close"})
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})
    shy = load_research("SHY")[["adj_close"]].rename(columns={"adj_close": "shy_close"})
    hyg = load_research("HYG")[["adj_close"]].rename(columns={"adj_close": "hyg_close"})
    lqd = load_research("LQD")[["adj_close"]].rename(columns={"adj_close": "lqd_close"})
    prices_aa = spy.join(tlt, how="left").join(gld, how="left").join(vix, how="left").join(shy, how="left").join(hyg, how="left").join(lqd, how="left")
    for col in ["tlt_close", "gld_close", "vix_close", "shy_close", "hyg_close", "lqd_close"]:
        prices_aa[col] = prices_aa[col].ffill()

    # --- Load stock panel ---
    print("Loading S&P 500 stock panel...")
    membership = load_sp500_members()
    tickers = get_all_historical_tickers()
    panel = load_stocks_panel(tickers)
    common_idx = panel.index.intersection(membership.index)
    panel = panel.loc[common_idx]; membership = membership.loc[common_idx]
    spy_rets = spy["adj_close"].pct_change().fillna(0.0)
    sectors = pd.read_parquet(DATA_DIR / "sp500_sectors.parquet")["sector"]
    pt = panel.columns.tolist()
    print(f"  Panel: {panel.shape[0]} days, {panel.shape[1]} tickers")

    # --- Load fundamentals ---
    print("Loading fundamentals...")
    ni = load_fundamentals_panel(pt, "net_income", panel.index)
    ocf = load_fundamentals_panel(pt, "operating_cf", panel.index)
    ta = load_fundamentals_panel(pt, "total_assets", panel.index)
    gp = load_fundamentals_panel(pt, "gross_profit", panel.index)
    rev = load_fundamentals_panel(pt, "revenue", panel.index)
    be = load_fundamentals_panel(pt, "book_equity", panel.index)
    ltd = load_fundamentals_panel(pt, "long_term_debt", panel.index)
    ca = load_fundamentals_panel(pt, "current_assets", panel.index)
    cl = load_fundamentals_panel(pt, "current_liabilities", panel.index)
    shares = load_fundamentals_panel(pt, "shares_outstanding", panel.index)

    folds_aa = make_folds(spy, train_years=5, test_years=2)
    folds_cs = make_folds(panel, train_years=5, test_years=2)
    print(f"Folds (AA): {len(folds_aa)}  |  Folds (CS): {len(folds_cs)}")

    component_returns: dict[str, pd.Series] = {}

    # --- H300/H350 core ETF anchor strategies ---
    aa_core = {
        "H021": RiskParity(),
        "H043": SellInMayBondConfirm(),
        "H078": SeasonalRiskParity(),
        "H046": HalloweenMomentumConfirm(),
        "H070": MultiAssetDynamicVolTarget(),
        "H086": LowVolTargetPortfolio(),
        "H087": RatchetDrawdownProtection(),
        "H097": CorrelationRiskParityHybrid(),
        "H060": CarryMomentumSPYTLT(),
        "H079": CrisisProtocolRiskParity(),
        "H035": GoldFearGauge(),
        "H301": VRPRiskParityScaled(),
        "H306": FundingStressMonitor(),
        "H308": CreditRegimeRiskParity(),
    }
    print(f"\nRunning {len(aa_core)} core ETF strategies...")
    for name, strat in aa_core.items():
        if isinstance(strat, MultiAssetStrategy):
            wf = walk_forward_multi(prices_aa, lambda _, s=strat: s, folds=folds_aa)
        else:
            wf = walk_forward(prices_aa, lambda _, s=strat: s, folds=folds_aa)
        component_returns[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    # --- Set 6 CS strategies (20 best, batches 1-4 + 6) ---
    cs_factories = {
        # Batch 1
        "H401": lambda tp, tm: AccrualQualityPort(net_income=ni, operating_cf=ocf, total_assets=ta, sectors=sectors),
        "H402": lambda tp, tm: EarningsMomentumQuality(net_income=ni, operating_cf=ocf, total_assets=ta, sectors=sectors),
        "H403": lambda tp, tm: TorpedoRiskAvoid(book_equity=be, net_income=ni, total_assets=ta, shares_outstanding=shares, sectors=sectors),
        "H404": lambda tp, tm: RevenueEarningsAcceleration(revenue=rev, net_income=ni, total_assets=ta, sectors=sectors),
        "H405": lambda tp, tm: CashEarningsPersistence(net_income=ni, operating_cf=ocf, sectors=sectors),
        # Batch 2
        "H408": lambda tp, tm: BookValueGrowthPort(book_equity=be, sectors=sectors),
        "H409": lambda tp, tm: BalanceSheetStrengthPort(book_equity=be, long_term_debt=ltd, operating_cf=ocf, total_assets=ta, sectors=sectors),
        # Batch 3 stars
        "H412": lambda tp, tm: FundamentalDivergenceAvoid(net_income=ni, total_assets=ta, sectors=sectors),
        "H415": lambda tp, tm: StableMarginMomentum(gross_profit=gp, revenue=rev, sectors=sectors),
        "H411": lambda tp, tm: ConsistentGrowthStreak(revenue=rev, net_income=ni, total_assets=ta, sectors=sectors),
        # Batch 4
        "H416": lambda tp, tm: DifferentiatedMarginPort(gross_profit=gp, revenue=rev, sectors=sectors),
        "H417": lambda tp, tm: CompetitivePressureAvoid(gross_profit=gp, revenue=rev, total_assets=ta, sectors=sectors),
        "H419": lambda tp, tm: CashDefenseProxy(gross_profit=gp, revenue=rev, total_assets=ta, current_assets=ca, sectors=sectors),
        "H420": lambda tp, tm: MarketShareProxy(revenue=rev, sectors=sectors),
        # Batch 6 star
        "H427": lambda tp, tm: CapitalProductivityPort(gross_profit=gp, total_assets=ta, sectors=sectors),
        "H428": lambda tp, tm: GrowthReinvestmentQuality(gross_profit=gp, total_assets=ta, revenue=rev, book_equity=be, sectors=sectors),
        "H429": lambda tp, tm: SectorRelativeMarginProxy(gross_profit=gp, revenue=rev, sectors=sectors),
    }

    print(f"\nRunning {len(cs_factories)} Set 6 CS strategies...")
    for name, factory in cs_factories.items():
        print(f"  {name}...", end="", flush=True)
        wf = walk_forward_cross_sectional(panel, membership, factory, folds=folds_cs, spy_returns=spy_rets)
        component_returns[name] = wf.combined_returns
        print(f" Sharpe {sharpe(wf.combined_returns):.3f}")

    # --- MDR optimisation ---
    returns_df = pd.DataFrame(component_returns)
    all_names = list(returns_df.columns)
    ew_rets = returns_df.mean(axis=1)
    print(f"\nEqual-weight ({len(all_names)} strategies): Sharpe {sharpe(ew_rets):.3f}  MaxDD {max_drawdown(ew_rets):.2%}")

    print("\nSolving MDR optimisation...")
    w_mdr = max_diversification_weights(returns_df)

    print("\nOptimal MDR weights (> 0.5%):")
    for name, w in sorted(zip(all_names, w_mdr, strict=True), key=lambda x: x[1], reverse=True):
        if w > 0.005:
            print(f"  {name}: {w:.4f}")

    mdr_rets = pd.Series(returns_df.values @ w_mdr, index=returns_df.index)
    lo, hi = bootstrap_sharpe_ci(mdr_rets, n_boot=5000)
    s = sharpe(mdr_rets)
    c = cagr(mdr_rets)
    dd = max_drawdown(mdr_rets)

    print(f"\nUltimate Set 6 MDR (H449):")
    print(f"  Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {c:.2%}  MaxDD {dd:.2%}")
    print(f"\n  Compare H300/H350: Sharpe 1.525  MaxDD -6.86%")

    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_aa)
    corr = float(mdr_rets.corr(wf_h001.combined_returns.reindex(mdr_rets.index)))
    print(f"\nH001 correlation: {corr:+.4f}")

    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds_aa):
        mask = (mdr_rets.index >= fold.test_start) & (mdr_rets.index < fold.test_end)
        fs_val = float(sharpe(mdr_rets[mask])) if mask.sum() > 0 else 0.0
        fold_sharpes.append(fs_val)
        print(f"  Fold {i:>2}: {fs_val:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    decision = "KEEP" if s > 0.51 and lo > 0 else "kill"
    print(f"\nDecision: {decision}")

    top_weights = {n: float(w) for n, w in zip(all_names, w_mdr, strict=True) if w > 0.005}
    write_summary(
        hypothesis_id="449", name="ultimate-set6-mdr", strategy_class="UltimateSet6MDR",
        family="Set6 meta-ensemble", sharpe=s, ci_low=lo, ci_high=hi, cagr=c,
        max_drawdown=dd, n_trades=0, h001_corr=corr, fold_sharpes=fold_sharpes,
        decision=decision,
        notes=(
            f"SLSQP MDR over {len(all_names)} strategies: {len(aa_core)} H300/H350 ETF core "
            f"+ {len(cs_factories)} Set 6 CS. Target: beat H300/H350 (Sharpe 1.525). "
            f"Top weights: {top_weights}."
        ),
        canonical_params={
            "n_strategies": len(all_names),
            "n_aa_core": len(aa_core),
            "n_cs_set6": len(cs_factories),
            "method": "SLSQP",
            "top_weights": top_weights,
        },
    )


if __name__ == "__main__":
    main()
