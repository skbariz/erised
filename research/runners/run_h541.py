"""H541: Set7EqualWeightCSEnsemble -- EW of top 5 Sharpe Set7 individual CS all-fold strategies.
H538/H540/H537/H535/H469. Baseline for Set7 CS contribution before MDR.
Run: uv run python scripts/run_h541.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from write_summary import write_summary
from erised.data.fundamentals import load_fundamentals_panel
from erised.data.loader import load_research
from erised.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
from erised.metrics.performance import bootstrap_sharpe_ci, cagr, max_drawdown, sharpe
from erised.strategies.base import MovingAverageCrossover
from erised.strategies.cross_sectional import (
    FundamentalReversalWeekly, LowBetaQualityMomentum, MinReturnMomentumCombo,
    MomentumWithPortfolioRatchet, ReversionQualityCombo,
)
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEP = "=" * 64

def main():
    print(f"H541: Set7EqualWeightCSEnsemble (top 5 Set7 CS all-fold, EW)\n{SEP}")
    tickers = get_all_historical_tickers(); panel = load_stocks_panel(tickers)
    membership = load_sp500_members(); common = panel.index.intersection(membership.index)
    panel = panel.loc[common]; membership = membership.loc[common]
    spy = load_research("SPY"); spy_rets = spy["adj_close"].pct_change().fillna(0.0)
    sectors = pd.read_parquet(DATA_DIR / "sp500_sectors.parquet")["sector"]
    pt = panel.columns.tolist()
    gp = load_fundamentals_panel(pt, "gross_profit", panel.index)
    ta = load_fundamentals_panel(pt, "total_assets", panel.index)
    ni = load_fundamentals_panel(pt, "net_income", panel.index)
    ocf = load_fundamentals_panel(pt, "operating_cf", panel.index)
    folds = make_folds(panel, train_years=5, test_years=2)
    print(f"Panel: {panel.shape[0]} days, {panel.shape[1]} tickers  Folds: {len(folds)}")

    # Top 5 Set7 individual CS all-fold strategies by OOS Sharpe
    cs_factories = {
        "H538": lambda tp, tm: ReversionQualityCombo(gross_profit=gp, total_assets=ta, sectors=sectors),
        "H540": lambda tp, tm: FundamentalReversalWeekly(net_income=ni, operating_cf=ocf, total_assets=ta, sectors=sectors),
        "H537": lambda tp, tm: MomentumWithPortfolioRatchet(sectors=sectors),
        "H535": lambda tp, tm: LowBetaQualityMomentum(gross_profit=gp, total_assets=ta, spy_returns=spy_rets, sectors=sectors),
        "H469": lambda tp, tm: MinReturnMomentumCombo(sectors=sectors),
    }
    all_rets = {}
    for name, factory in cs_factories.items():
        wf = walk_forward_cross_sectional(panel, membership, factory, folds=folds, spy_returns=spy_rets)
        all_rets[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}")

    ret_df = pd.DataFrame(all_rets)
    ew = ret_df.mean(axis=1)
    lo, hi = bootstrap_sharpe_ci(ew, n_boot=5000)
    s = sharpe(ew); c = cagr(ew); dd = max_drawdown(ew)
    print(f"\nEqual-weight: Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]  CAGR {c:.2%}  MaxDD {dd:.2%}")
    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds):
        mask = (ew.index >= fold.test_start) & (ew.index < fold.test_end)
        fr = ew[mask]; fs = sharpe(fr) if len(fr) > 10 else 0.0
        fold_sharpes.append(float(fs)); print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")
    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=make_folds(spy, train_years=5, test_years=2))
    corr = ew.corr(wh.combined_returns.reindex(ew.index))
    print(f"\nH001 correlation: {corr:+.4f}"); decision = "KEEP" if s > 0.51 and lo > 0 else "kill"; print(f"Decision: {decision}")
    write_summary(hypothesis_id="541", name="set7-equal-weight-cs-ensemble", strategy_class="Set7EqualWeightCSEnsemble",
        family="Set7 Batch10 Terminal Ensemble", sharpe=s, ci_low=lo, ci_high=hi, cagr=c,
        max_drawdown=dd, n_trades=0, h001_corr=float(corr), fold_sharpes=fold_sharpes, decision=decision,
        notes="Equal-weight of top 5 Set7 individual CS all-fold strategies by OOS Sharpe: H538 ReversionQualityCombo, H540 FundamentalReversalWeekly, H537 MomentumWithPortfolioRatchet, H535 LowBetaQualityMomentum, H469 MinReturnMomentumCombo. Baseline before MDR.",
        canonical_params={"components": list(cs_factories.keys()), "method": "equal_weight"})

if __name__ == "__main__": main()
