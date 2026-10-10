"""H299: Set3RiskAvoidanceMDR -- MDR ensemble over best 5 H251-H298 strategies.

Run: uv run python scripts/run_h299.py
Expected runtime: ~45 minutes (5 cross-sectional walk-forwards + MDR).
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
from erised.strategies.base import MovingAverageCrossover
from erised.strategies.cross_sectional import (
    InvestmentEfficiencyRatio,
    KnowledgeReturnProxy,
    LowCoverageMomentumProxy,
    SustainabilityFilteredProfitability,
    TailRiskComposite,
)
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEP = "=" * 64


def max_diversification_weights(returns_df: pd.DataFrame) -> np.ndarray:
    """SLSQP: maximise DR = sum(w*sigma) / sqrt(w^T Cov w). sum(w)=1, w>=0."""
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


def main() -> None:
    print(f"H299: Set3RiskAvoidanceMDR (MDR over 5 best H251-H298)\n{SEP}")

    membership = load_sp500_members()
    tickers = get_all_historical_tickers()
    print(f"Loading prices for {len(tickers)} historical tickers...")
    panel = load_stocks_panel(tickers)

    common_idx = panel.index.intersection(membership.index)
    panel = panel.loc[common_idx]
    membership = membership.loc[common_idx]

    spy = load_research("SPY")
    spy_rets = spy["adj_close"].pct_change().fillna(0.0)

    sectors = pd.read_parquet(DATA_DIR / "sp500_sectors.parquet")["sector"]
    panel_tickers = panel.columns.tolist()

    print("Loading fundamental data...")
    gp = load_fundamentals_panel(panel_tickers, "gross_profit", panel.index)
    ta = load_fundamentals_panel(panel_tickers, "total_assets", panel.index)
    ca = load_fundamentals_panel(panel_tickers, "current_assets", panel.index)
    cl = load_fundamentals_panel(panel_tickers, "current_liabilities", panel.index)
    ni = load_fundamentals_panel(panel_tickers, "net_income", panel.index)
    ocf = load_fundamentals_panel(panel_tickers, "operating_cf", panel.index)
    rev = load_fundamentals_panel(panel_tickers, "revenue", panel.index)

    folds = make_folds(panel, train_years=5, test_years=2)
    print(f"Data: {panel.shape[0]} days, {panel.shape[1]} tickers")
    print(f"Folds: {len(folds)}")

    # ------------------------------------------------------------------
    # Selected candidates (5 strategies from different Set 3 families):
    # H268: KnowledgeReturnProxy -- GP/TA top-decile (0.922, best quality)
    # H265: SustainabilityFilteredProfitability -- GP/TA + accruals filter (0.952)
    # H254: TailRiskComposite -- tail risk avoidance
    # H287: InvestmentEfficiencyRatio -- capital discipline (0.790)
    # H284: LowCoverageMomentumProxy -- momentum with idio-vol adjustment (0.974)
    # ------------------------------------------------------------------

    cs_factories = {
        "H268": lambda tp, tm: KnowledgeReturnProxy(
            gross_profit=gp, total_assets=ta, sectors=sectors
        ),
        "H265": lambda tp, tm: SustainabilityFilteredProfitability(
            gross_profit=gp,
            total_assets=ta,
            current_assets=ca,
            current_liabilities=cl,
            net_income=ni,
            operating_cf=ocf,
            sectors=sectors,
        ),
        "H254": lambda tp, tm: TailRiskComposite(sectors=sectors, spy_rets=spy_rets),
        "H287": lambda tp, tm: InvestmentEfficiencyRatio(
            revenue=rev,
            total_assets=ta,
            current_assets=ca,
            sectors=sectors,
        ),
        "H284": lambda tp, tm: LowCoverageMomentumProxy(
            spy_prices=spy["adj_close"], sectors=sectors
        ),
    }

    print(f"\nRunning {len(cs_factories)} Set 3 strategies...")
    component_returns = {}
    for name, factory in cs_factories.items():
        print(f"  {name}...", end="", flush=True)
        wf = walk_forward_cross_sectional(
            panel, membership, factory, folds=folds, spy_returns=spy_rets
        )
        component_returns[name] = wf.combined_returns
        print(f" Sharpe {sharpe(wf.combined_returns):.3f}")

    returns_df = pd.DataFrame(component_returns)

    # Equal-weight baseline
    ew_rets = returns_df.mean(axis=1)
    print(
        f"\nEqual-weight baseline: Sharpe {sharpe(ew_rets):.3f}  MaxDD {max_drawdown(ew_rets):.2%}"
    )

    # MDR optimization
    print("\nSolving MDR optimisation...")
    all_names = list(returns_df.columns)
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

    print("\nSet3 MDR Portfolio:")
    print(f"  Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    print(f"  CAGR {c:.2%}  MaxDD {dd:.2%}")

    # H001 correlation
    folds_spy = make_folds(spy, train_years=5, test_years=2)
    wf_h001 = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=folds_spy)
    corr = mdr_rets.corr(wf_h001.combined_returns.reindex(mdr_rets.index))
    print(f"\nH001 correlation: {corr:+.4f}")

    # Per-fold Sharpe
    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds):
        mask = (mdr_rets.index >= fold.test_start) & (mdr_rets.index < fold.test_end)
        fold_ret = mdr_rets[mask]
        fs = sharpe(fold_ret) if len(fold_ret) > 0 else 0.0
        fold_sharpes.append(fs)
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    decision = "KEEP" if s > 0.51 and lo > 0 else "kill"
    print(f"\nDecision: {decision}")

    top_weights = {n: float(w) for n, w in zip(all_names, w_mdr, strict=True) if w > 0.005}
    write_summary(
        hypothesis_id="299",
        name="set3-risk-avoidance-mdr",
        strategy_class="Set3RiskAvoidanceMDR",
        family="ensemble",
        sharpe=s,
        ci_low=lo,
        ci_high=hi,
        cagr=c,
        max_drawdown=dd,
        n_trades=0,
        h001_corr=float(corr),
        fold_sharpes=[float(fs) for fs in fold_sharpes],
        decision=decision,
        notes=(
            f"SLSQP MDR over 5 Set 3 strategies: H268 (GP/TA decile), H265 (sustainability GP), "
            f"H254 (tail risk), H287 (investment efficiency), H284 (coverage-adjusted momentum). "
            f"Top weights: {top_weights}. "
            f"Equal-weight baseline: Sharpe {sharpe(ew_rets):.3f}."
        ),
        canonical_params={
            "n_strategies": len(cs_factories),
            "method": "SLSQP",
            "candidates": list(cs_factories.keys()),
            "top_weights": top_weights,
        },
    )


if __name__ == "__main__":
    main()
