"""H486: SectorFundamentalRotation -- H021 with cross-sector GP/TA dispersion signal.
High sector quality dispersion → equity tilt × 1.10; low → × 0.90.
Run: uv run python scripts/run_h486.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from write_summary import write_summary
from erised.data.fundamentals import load_fundamentals_panel
from erised.data.loader import load_research
from erised.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
from erised.metrics.performance import bootstrap_sharpe_ci, sharpe
from erised.strategies import SectorFundamentalRotation, MovingAverageCrossover
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_multi
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEP = "=" * 64

def main():
    print(f"H486: SectorFundamentalRotation (cross-sector GP/TA dispersion overlay)\n{SEP}")
    tickers = get_all_historical_tickers()
    panel = load_stocks_panel(tickers)
    membership = load_sp500_members(); common = panel.index.intersection(membership.index)
    panel = panel.loc[common]; pt = panel.columns.tolist()
    sectors = pd.read_parquet(DATA_DIR / "sp500_sectors.parquet")["sector"]
    gp = load_fundamentals_panel(pt, "gross_profit", panel.index)
    ta = load_fundamentals_panel(pt, "total_assets", panel.index)
    ta_safe = ta.where(ta > 0)
    gp_ta = (gp / ta_safe).clip(0.0, 5.0)
    sector_medians = []
    for sec in sectors.dropna().unique():
        tickers_sec = [t for t in sectors[sectors == sec].index if t in gp_ta.columns]
        if tickers_sec:
            sector_medians.append(gp_ta[tickers_sec].median(axis=1))
    sector_disp = pd.concat(sector_medians, axis=1).std(axis=1).rolling(63, min_periods=20).mean()

    spy = load_research("SPY"); tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    prices = spy.join(tlt, how="left"); prices["tlt_close"] = prices["tlt_close"].ffill()
    folds = make_folds(prices, train_years=5, test_years=2)
    print(f"Data: {len(prices)} rows  Folds: {len(folds)}")
    wf = walk_forward_multi(prices, lambda _: SectorFundamentalRotation(sector_quality_dispersion=sector_disp), folds=folds)
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
        hypothesis_id="486", name="sector-fundamental-rotation", strategy_class="SectorFundamentalRotation",
        family="Set7 Hybrid AA+CS", sharpe=rep.sharpe, ci_low=lo, ci_high=hi, cagr=rep.cagr,
        max_drawdown=rep.max_drawdown, n_trades=rep.n_trades, h001_corr=float(corr),
        fold_sharpes=[float(sharpe(f.net_returns)) for f in wf.fold_results], decision=decision,
        notes="H021 + cross-sector GP/TA quality dispersion. Rolling 63d std of per-sector median GP/TA. High dispersion (tech dominating) → SPY × 1.10; low → × 0.90. Novel sector-level fundamental signal.",
        canonical_params={"vol_lookback": 60, "signal_window": 252, "bull_scale": 1.10, "bear_scale": 0.90},
    )

if __name__ == "__main__":
    main()
