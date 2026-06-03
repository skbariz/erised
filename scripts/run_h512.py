"""H512: SeasonalCSQualityBridge -- Nov-Apr: H415 (StableMarginMomentum CS); May-Oct: 100% TLT.
Replaces H078's H021 Nov-Apr leg with the stronger all-fold CS quality signal.
Run: uv run python scripts/run_h512.py
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
from quant.metrics.performance import bootstrap_sharpe_ci, cagr, max_drawdown, sharpe
from quant.strategies.base import MovingAverageCrossover
from quant.strategies.cross_sectional import StableMarginMomentum
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEP = "=" * 64

def main():
    print(f"H512: SeasonalCSQualityBridge (Nov-Apr: H415 CS; May-Oct: 100% TLT)\n{SEP}")
    tickers = get_all_historical_tickers()
    panel = load_stocks_panel(tickers)
    membership = load_sp500_members(); common = panel.index.intersection(membership.index)
    panel = panel.loc[common]; membership = membership.loc[common]
    spy = load_research("SPY"); spy_rets = spy["adj_close"].pct_change().fillna(0.0)
    tlt = load_research("TLT")[["adj_close"]]; tlt_rets = tlt["adj_close"].pct_change()
    sectors = pd.read_parquet(DATA_DIR / "sp500_sectors.parquet")["sector"]
    pt = panel.columns.tolist()
    gp = load_fundamentals_panel(pt, "gross_profit", panel.index)
    rev = load_fundamentals_panel(pt, "revenue", panel.index)
    folds = make_folds(panel, train_years=5, test_years=2)
    print(f"Data: {panel.shape[0]} days, {panel.shape[1]} tickers")

    wf_h415 = walk_forward_cross_sectional(
        panel, membership,
        lambda tp, tm: StableMarginMomentum(gross_profit=gp, revenue=rev, sectors=sectors),
        folds=folds, spy_returns=spy_rets,
    )
    r_h415 = wf_h415.combined_returns
    print(f"H415 (StableMarginMomentum): Sharpe {sharpe(r_h415):.3f}  MaxDD {max_drawdown(r_h415):.2%}")

    # Seasonal mask: Nov-Apr = H415 returns; May-Oct = TLT returns
    in_season = pd.Series(r_h415.index.month.isin([11, 12, 1, 2, 3, 4]), index=r_h415.index)
    tlt_aligned = tlt_rets.reindex(r_h415.index).fillna(0.0)
    blended = r_h415 * in_season.astype(float) + tlt_aligned * (~in_season).astype(float)

    lo, hi = bootstrap_sharpe_ci(blended, n_boot=5000)
    s = sharpe(blended); c = cagr(blended); dd = max_drawdown(blended)
    print(f"\nSeasonal blend: Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]  CAGR {c:.2%}  MaxDD {dd:.2%}")
    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds):
        mask = (blended.index >= fold.test_start) & (blended.index < fold.test_end)
        fr = blended[mask]; fs = sharpe(fr) if len(fr) > 10 else 0.0
        fold_sharpes.append(float(fs))
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    fs_spy = make_folds(spy, train_years=5, test_years=2)
    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=fs_spy)
    corr = blended.corr(wh.combined_returns.reindex(blended.index))
    print(f"\nH001 correlation: {corr:+.4f}"); decision = "KEEP" if s > 0.51 and lo > 0 else "kill"; print(f"Decision: {decision}")
    write_summary(
        hypothesis_id="512", name="seasonal-cs-quality-bridge", strategy_class="SeasonalCSQualityBridge",
        family="Set7 Cross-Program Synergy", sharpe=s, ci_low=lo, ci_high=hi, cagr=c,
        max_drawdown=dd, n_trades=wf_h415.combined_report.n_trades, h001_corr=float(corr),
        fold_sharpes=fold_sharpes, decision=decision,
        notes="H078's seasonal structure with H021 replaced by H415 (StableMarginMomentum CS) in Nov-Apr. May-Oct stays 100% TLT (H078's defensive leg). Tests if H415 CS is more powerful than H021 AA in the favorable equity season.",
        canonical_params={"cs_strategy": "H415_StableMarginMomentum", "favorable_months": [11, 12, 1, 2, 3, 4]},
    )

if __name__ == "__main__":
    main()
