"""H510: CrossProgramHybridLowTurnover -- 80% H021 (daily) + 20% H415 (quarterly CS rebalance).
Explicit liquidity-aware frequency split. ETFs daily; stocks quarterly. 80/20 vs H490's 50/50.
Run: uv run python scripts/run_h510.py
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
from quant.strategies.base import MovingAverageCrossover, RiskParity
from quant.strategies.cross_sectional import StableMarginMomentum
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional, walk_forward_multi
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEP = "=" * 64

def main():
    print(f"H510: CrossProgramHybridLowTurnover (80% H021 daily + 20% H415 quarterly)\n{SEP}")
    # AA leg: H021 (RiskParity, daily)
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    prices_aa = spy.join(tlt, how="left"); prices_aa["tlt_close"] = prices_aa["tlt_close"].ffill()
    folds_aa = make_folds(prices_aa, train_years=5, test_years=2)
    wf_h021 = walk_forward_multi(prices_aa, lambda _: RiskParity(60), folds=folds_aa)
    print(f"H021 (RiskParity):          Sharpe {sharpe(wf_h021.combined_returns):.3f}  MaxDD {max_drawdown(wf_h021.combined_returns):.2%}")

    # CS leg: H415 (StableMarginMomentum, quarterly rebalance=63d)
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
    wf_h415q = walk_forward_cross_sectional(
        panel, membership,
        lambda tp, tm: StableMarginMomentum(gross_profit=gp, revenue=rev, sectors=sectors, rebalance_period=63),
        folds=folds_cs, spy_returns=spy_rets,
    )
    print(f"H415q (StableMarginMom 63d): Sharpe {sharpe(wf_h415q.combined_returns):.3f}  MaxDD {max_drawdown(wf_h415q.combined_returns):.2%}")

    # Blend 80% H021 + 20% H415 quarterly
    r021 = wf_h021.combined_returns; r415q = wf_h415q.combined_returns
    common_idx = r021.index.intersection(r415q.index)
    blended = r021.loc[common_idx] * 0.80 + r415q.loc[common_idx] * 0.20
    lo, hi = bootstrap_sharpe_ci(blended, n_boot=5000)
    s = sharpe(blended); c = cagr(blended); dd = max_drawdown(blended)
    print(f"\n80/20 Blend: Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]  CAGR {c:.2%}  MaxDD {dd:.2%}")

    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds_aa):
        mask = (blended.index >= fold.test_start) & (blended.index < fold.test_end)
        fold_ret = blended[mask]
        fs = sharpe(fold_ret) if len(fold_ret) > 10 else 0.0
        fold_sharpes.append(float(fs))
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    fs_spy = make_folds(spy, train_years=5, test_years=2)
    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=fs_spy)
    corr = blended.corr(wh.combined_returns.reindex(blended.index))
    print(f"\nH001 correlation: {corr:+.4f}"); decision = "KEEP" if s > 0.51 and lo > 0 else "kill"; print(f"Decision: {decision}")

    n_trades = wf_h021.combined_report.n_trades + wf_h415q.combined_report.n_trades
    write_summary(
        hypothesis_id="510", name="cross-program-hybrid-low-turnover", strategy_class="CrossProgramHybridLowTurnover",
        family="Set7 Execution-Aware", sharpe=s, ci_low=lo, ci_high=hi, cagr=c,
        max_drawdown=dd, n_trades=n_trades, h001_corr=float(corr),
        fold_sharpes=fold_sharpes, decision=decision,
        notes="80% H021 (RiskParity, daily rebalance) + 20% H415 (StableMarginMomentum, quarterly 63d rebalance). Explicit frequency-by-asset-class split: ETFs daily, stocks quarterly. H490 tested 50/50 daily; H510 tests 80/20 with quarterly CS.",
        canonical_params={"blend_aa": 0.80, "blend_cs": 0.20, "h021_vol_lookback": 60, "h415_rebalance_period": 63},
    )

if __name__ == "__main__":
    main()
