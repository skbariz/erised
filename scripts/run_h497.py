"""H497: RegimeAdaptiveCSSignal -- VIX-gated CS signal: momentum (VIX<20) or quality (VIX>=20).
Sector-neutral, monthly. Tests momentum-calm / quality-stress academic hypothesis.
Run: uv run python scripts/run_h497.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from write_summary import write_summary
from quant.data.fundamentals import load_fundamentals_panel
from quant.data.loader import load_research
from quant.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
from quant.metrics.performance import bootstrap_sharpe_ci, sharpe
from quant.strategies.base import MovingAverageCrossover
from quant.strategies.cross_sectional import RegimeAdaptiveCSSignal
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEP = "=" * 64

def main():
    print(f"H497: RegimeAdaptiveCSSignal (VIX<20→momentum; VIX>=20→quality, monthly)\n{SEP}")
    tickers = get_all_historical_tickers()
    panel = load_stocks_panel(tickers)
    membership = load_sp500_members(); common = panel.index.intersection(membership.index)
    panel = panel.loc[common]; membership = membership.loc[common]
    spy = load_research("SPY"); spy_rets = spy["adj_close"].pct_change().fillna(0.0)
    sectors = pd.read_parquet(DATA_DIR / "sp500_sectors.parquet")["sector"]
    pt = panel.columns.tolist()
    gp = load_fundamentals_panel(pt, "gross_profit", panel.index)
    ta = load_fundamentals_panel(pt, "total_assets", panel.index)
    vix = pd.read_parquet(DATA_DIR / "VIX_research.parquet")[["adj_close"]].rename(columns={"adj_close": "vix_close"})["vix_close"]
    folds = make_folds(panel, train_years=5, test_years=2)
    print(f"Data: {panel.shape[0]} days, {panel.shape[1]} tickers  Folds: {len(folds)}")
    wf = walk_forward_cross_sectional(
        panel, membership,
        lambda tp, tm: RegimeAdaptiveCSSignal(sectors=sectors, gross_profit=gp, total_assets=ta, vix_series=vix),
        folds=folds, spy_returns=spy_rets,
    )
    rep = wf.combined_report; lo, hi = bootstrap_sharpe_ci(wf.combined_returns, n_boot=5000)
    print(f"\nCanonical: Sharpe {rep.sharpe:.3f}  CI [{lo:.3f}, {hi:.3f}]  CAGR {rep.cagr:.2%}  MaxDD {rep.max_drawdown:.2%}")
    print("\nPer-fold Sharpe:")
    for i, f in enumerate(wf.fold_results):
        print(f"  Fold {i:>2}: {sharpe(f.net_returns):+.3f}  ({folds[i].test_start.date()} -- {folds[i].test_end.date()})")
    fs = make_folds(spy, train_years=5, test_years=2)
    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=fs)
    corr = wf.combined_returns.corr(wh.combined_returns.reindex(wf.combined_returns.index))
    print(f"\nH001 correlation: {corr:+.4f}"); decision = "KEEP" if rep.sharpe > 0.51 and lo > 0 else "kill"; print(f"Decision: {decision}")
    write_summary(
        hypothesis_id="497", name="regime-adaptive-cs-signal", strategy_class="RegimeAdaptiveCSSignal",
        family="Set7 Continuous Macro AA", sharpe=rep.sharpe, ci_low=lo, ci_high=hi, cagr=rep.cagr,
        max_drawdown=rep.max_drawdown, n_trades=rep.n_trades, h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results], decision=decision,
        notes="VIX-conditional CS signal selection. VIX < 20 → 6m price momentum (top 20%); VIX >= 20 → GP/TA quality (top 20%). Sector-neutral, monthly. Quality mode uses fundamentals (EDGAR gap in folds 0-5). Tests momentum-in-calm / quality-in-stress hypothesis.",
        canonical_params={"vix_threshold": 20.0, "mom_lookback": 126, "within_sector_quantile": 0.20, "rebalance_period": 21},
    )

if __name__ == "__main__":
    main()
