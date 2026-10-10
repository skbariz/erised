"""H346: Set4ETFMDRPortfolio -- MDR over 28 kept H301-H340 ETF strategies.

All 28 kept individual H301-H340 ETF strategies (excludes H321-H325 ensembles and
H341-H345 runner-level sleeve blends). SLSQP Max-Diversification Ratio optimisation.
Benchmark for whether Set 4 strategies are internally diversified enough to benefit
from MDR combination. Compare to H095 (Set 1 MDR: Sharpe 1.133).

Run: uv run python scripts/run_h346.py
Expected runtime: ~30 minutes.
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

from erised.data.loader import load_research
from erised.metrics.performance import bootstrap_sharpe_ci, cagr, max_drawdown, sharpe
from erised.strategies import (
    ConstrainedMacroOverlay,
    CreditRegimeRiskParity,
    DefaultSpreadSPYTimer,
    DefensiveMacroStateMachine,
    DoubleCheckH080,
    DualFrequencyVolTarget,
    EarningsYieldOverlay,
    FundingStressH077,
    FundingStressMonitor,
    FundingStressVolTarget,
    LiquiditySpiralProtection,
    MacroCombinationOverlay,
    MacroStateH089,
    MovingAverageCrossover,
    MultiAssetStrategy,
    MultiSignalStressAdjusted,
    QuarterlyMacroStateOverlay,
    RatchetVolTarget,
    RegimeClassifier,
    RiskBudgetExpansion,
    SimpleVRPOverlay,
    TermSpreadSPYTimer,
    ValuationSpreadMarketState,
    VolAccelerationDetector,
    VolPercentileExposure,
    VolRegimeRiskParity,
    VRPCorrelationRegime,
    VRPRiskParityScaled,
    VRPTrendConfirm,
    VRPVolTarget,
)
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi

SEP = "=" * 64

H301_H340_STRATEGIES = {
    "H301": SimpleVRPOverlay(),
    "H302": VRPRiskParityScaled(),
    "H303": VRPTrendConfirm(),
    "H304": VRPVolTarget(),
    "H306": FundingStressMonitor(),
    "H307": LiquiditySpiralProtection(),
    "H308": CreditRegimeRiskParity(),
    "H309": MultiSignalStressAdjusted(),
    "H310": FundingStressVolTarget(),
    "H311": TermSpreadSPYTimer(),
    "H312": DefaultSpreadSPYTimer(),
    "H313": EarningsYieldOverlay(),
    "H314": MacroCombinationOverlay(),
    "H315": ConstrainedMacroOverlay(),
    "H316": QuarterlyMacroStateOverlay(),
    "H317": ValuationSpreadMarketState(),
    "H318": RegimeClassifier(),
    "H319": RiskBudgetExpansion(),
    "H320": DefensiveMacroStateMachine(),
    "H326": DualFrequencyVolTarget(),
    "H327": VolAccelerationDetector(),
    "H328": VolPercentileExposure(),
    "H329": VolRegimeRiskParity(),
    "H330": RatchetVolTarget(),
    "H335": MacroStateH089(),
    "H337": FundingStressH077(),
    "H338": DoubleCheckH080(),
    "H339": VRPCorrelationRegime(),
}


def run_wf(prices, strat, folds):
    if isinstance(strat, MultiAssetStrategy):
        return walk_forward_multi(prices, lambda _, s=strat: s, folds=folds)
    return walk_forward(prices, lambda _, s=strat: s, folds=folds)


def mdr_weights(returns_df: pd.DataFrame) -> np.ndarray:
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
        neg_dr,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return result.x


def main():
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    gld = load_research("GLD")[["adj_close"]].rename(columns={"adj_close": "gld_close"})
    vix = load_research("VIX")[["adj_close"]].rename(columns={"adj_close": "vix_close"})
    shy = load_research("SHY")[["adj_close"]].rename(columns={"adj_close": "shy_close"})
    dbc = load_research("DBC")[["adj_close"]].rename(columns={"adj_close": "dbc_close"})
    xlp = load_research("XLP")[["adj_close"]].rename(columns={"adj_close": "xlp_close"})
    xlu = load_research("XLU")[["adj_close"]].rename(columns={"adj_close": "xlu_close"})
    ive = load_research("IVE")[["adj_close"]].rename(columns={"adj_close": "ive_close"})
    hyg = load_research("HYG")[["adj_close"]].rename(columns={"adj_close": "hyg_close"})
    lqd = load_research("LQD")[["adj_close"]].rename(columns={"adj_close": "lqd_close"})
    ivw = load_research("IVW")[["adj_close"]].rename(columns={"adj_close": "ivw_close"})

    prices_aa = (
        spy.join(tlt, how="left")
        .join(gld, how="left")
        .join(vix, how="left")
        .join(shy, how="left")
        .join(dbc, how="left")
        .join(xlp, how="left")
        .join(xlu, how="left")
        .join(ive, how="left")
        .join(hyg, how="left")
        .join(lqd, how="left")
        .join(ivw, how="left")
    )
    for col in [
        "tlt_close", "gld_close", "vix_close", "shy_close", "dbc_close",
        "xlp_close", "xlu_close", "ive_close", "hyg_close", "lqd_close", "ivw_close",
    ]:
        prices_aa[col] = prices_aa[col].ffill()

    folds = make_folds(spy, train_years=5, test_years=2)
    print(f"H346: Set4ETFMDRPortfolio ({len(H301_H340_STRATEGIES)} strategies)\n{SEP}")
    print(f"Data: {len(spy)} rows  {spy.index.min().date()} -- {spy.index.max().date()}")
    print(f"Folds: {len(folds)}")

    print("\nRunning all component strategies...")
    component_returns: dict[str, pd.Series] = {}
    for name, strat in H301_H340_STRATEGIES.items():
        wf = run_wf(prices_aa, strat, folds)
        component_returns[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    returns_df = pd.DataFrame(component_returns)
    ew_rets = returns_df.mean(axis=1)
    print(
        f"\nEqual-weight baseline ({len(H301_H340_STRATEGIES)} strategies):"
        f" Sharpe {sharpe(ew_rets):.3f}  MaxDD {max_drawdown(ew_rets):.2%}"
    )

    print("\nSolving MDR optimisation...")
    w_mdr = mdr_weights(returns_df)
    names = list(H301_H340_STRATEGIES.keys())

    print("\nOptimal MDR weights (> 0.5%):")
    for nm, w in sorted(zip(names, w_mdr, strict=True), key=lambda x: x[1], reverse=True):
        if w > 0.005:
            print(f"  {nm}: {w:.4f}")

    mdr_rets = pd.Series(returns_df.values @ w_mdr, index=returns_df.index)
    lo, hi = bootstrap_sharpe_ci(mdr_rets, n_boot=5000)
    s = sharpe(mdr_rets)
    c = cagr(mdr_rets)
    dd = max_drawdown(mdr_rets)

    print("\nSet4 MDR Portfolio (H346):")
    print(f"  Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {c:.2%}  MaxDD {dd:.2%}")
    print("\n  Compare H095 (Set1 MDR):  Sharpe 1.133  MaxDD -7.16%")

    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds)
    corr = float(mdr_rets.corr(wf_h001.combined_returns.reindex(mdr_rets.index)))
    print(f"\nH001 correlation: {corr:+.4f}")

    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds):
        fold_rets = mdr_rets.loc[fold.test_start : fold.test_end]
        fs = float(sharpe(fold_rets)) if len(fold_rets) > 0 else 0.0
        fold_sharpes.append(fs)
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    decision = "KEEP" if lo > 0 and s > 0.51 else "kill"
    print(f"Decision: {decision}")

    top_weights = {nm: float(w) for nm, w in zip(names, w_mdr, strict=True) if w > 0.005}
    write_summary(
        hypothesis_id="346",
        name="set4-etf-mdr-portfolio",
        strategy_class="Set4ETFMDRPortfolio",
        family="ensemble",
        sharpe=s,
        ci_low=lo,
        ci_high=hi,
        cagr=c,
        max_drawdown=dd,
        n_trades=0,
        h001_corr=corr,
        fold_sharpes=fold_sharpes,
        decision=decision,
        notes=(
            f"SLSQP MDR over {len(H301_H340_STRATEGIES)} kept H301-H340 ETF strategies "
            "(excludes H321-H325 ensembles and H341-H345 sleeve blends). "
            f"EW baseline: Sharpe {sharpe(ew_rets):.3f}. "
            f"Top MDR weights: {top_weights}."
        ),
        canonical_params={"method": "MDR_SLSQP", "n_strategies": len(H301_H340_STRATEGIES)},
    )


if __name__ == "__main__":
    main()
