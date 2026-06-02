"""H490: CrossProgramMidTermBlend -- 50% H086 (LowVolTarget AA) + 50% H415 (StableMarginMomentum CS).
Near-orthogonal H001 corrs: H086 +0.413, H415 +0.002. Blend hypothesis: CS lifts Sharpe, AA controls MaxDD.
Run: uv run python scripts/run_h490.py
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
from quant.strategies.base import LowVolTargetPortfolio, MovingAverageCrossover
from quant.strategies.cross_sectional import StableMarginMomentum
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional, walk_forward_multi
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEP = "=" * 64

def main():
    print(f"H490: CrossProgramMidTermBlend (50% H086 + 50% H415)\n{SEP}")
    # H086 setup
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    prices_aa = spy.join(tlt, how="left"); prices_aa["tlt_close"] = prices_aa["tlt_close"].ffill()
    folds_aa = make_folds(prices_aa, train_years=5, test_years=2)
    wf_h086 = walk_forward_multi(prices_aa, lambda _: LowVolTargetPortfolio(), folds=folds_aa)
    print(f"H086 (LowVolTarget): Sharpe {sharpe(wf_h086.combined_returns):.3f}  MaxDD {max_drawdown(wf_h086.combined_returns):.2%}")

    # H415 setup
    tickers = get_all_historical_tickers()
    panel = load_stocks_panel(tickers)
    membership = load_sp500_members(); common = panel.index.intersection(membership.index)
    panel = panel.loc[common]; membership = membership.loc[common]
    spy_rets = spy["adj_close"].pct_change().fillna(0.0)
    sectors = pd.read_parquet(DATA_DIR / "sp500_sectors.parquet")["sector"]
    folds_cs = make_folds(panel, train_years=5, test_years=2)
    pt = panel.columns.tolist()
    gp = load_fundamentals_panel(pt, "gross_profit", panel.index)
    rev = load_fundamentals_panel(pt, "revenue", panel.index)
    wf_h415 = walk_forward_cross_sectional(
        panel, membership,
        lambda tp, tm: StableMarginMomentum(gross_profit=gp, revenue=rev, sectors=sectors),
        folds=folds_cs, spy_returns=spy_rets,
    )
    print(f"H415 (StableMarginMom): Sharpe {sharpe(wf_h415.combined_returns):.3f}  MaxDD {max_drawdown(wf_h415.combined_returns):.2%}")

    # Blend 50/50
    r86 = wf_h086.combined_returns; r415 = wf_h415.combined_returns
    common_idx = r86.index.intersection(r415.index)
    blended = (r86.loc[common_idx] + r415.loc[common_idx]) / 2
    lo, hi = bootstrap_sharpe_ci(blended, n_boot=5000)
    s = sharpe(blended); c = cagr(blended); dd = max_drawdown(blended)
    print(f"\n50/50 Blend: Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]  CAGR {c:.2%}  MaxDD {dd:.2%}")

    print("\nPer-fold Sharpe:")
    folds = folds_aa
    fold_sharpes = []
    for i, fold in enumerate(folds):
        mask = (blended.index >= fold.test_start) & (blended.index < fold.test_end)
        fold_ret = blended[mask]
        fs = sharpe(fold_ret) if len(fold_ret) > 10 else 0.0
        fold_sharpes.append(float(fs))
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    fs_spy = make_folds(spy, train_years=5, test_years=2)
    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=fs_spy)
    corr = blended.corr(wh.combined_returns.reindex(blended.index))
    print(f"\nH001 correlation: {corr:+.4f}"); decision = "KEEP" if s > 0.51 and lo > 0 else "kill"; print(f"Decision: {decision}")

    # Approximate n_trades
    n_trades = wf_h086.combined_report.n_trades + wf_h415.combined_report.n_trades
    write_summary(
        hypothesis_id="490", name="cross-program-midterm-blend", strategy_class="CrossProgramMidTermBlend",
        family="Set7 Hybrid AA+CS", sharpe=s, ci_low=lo, ci_high=hi, cagr=c,
        max_drawdown=dd, n_trades=n_trades, h001_corr=float(corr),
        fold_sharpes=fold_sharpes, decision=decision,
        notes="50% H086 (LowVolTargetPortfolio, Sharpe 1.109, H001 +0.413) + 50% H415 (StableMarginMomentum, Sharpe 0.987, H001 +0.002). Near-orthogonal H001 corrs. CS lifts Sharpe; AA controls MaxDD.",
        canonical_params={"blend": 0.50, "h086_vol_target": 0.08, "h415_rebalance_period": 21},
    )

if __name__ == "__main__":
    main()
