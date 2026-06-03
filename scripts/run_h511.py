"""H511: VolScaledCSStarCombo -- equal-weight H412+H415+H429+H447 with portfolio-level vol-scaling.
H086's vol-scaling mechanism applied to CS star ensemble. Targets: MaxDD < -40%, Sharpe improvement.
Run: uv run python scripts/run_h511.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from write_summary import write_summary
from quant.data.fundamentals import load_fundamentals_panel
from quant.data.loader import load_research
from quant.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
from quant.metrics.performance import bootstrap_sharpe_ci, cagr, max_drawdown, sharpe
from quant.strategies.base import MovingAverageCrossover
from quant.strategies.cross_sectional import (
    CreditQualityMomentumCombo, FundamentalDivergenceAvoid,
    SectorRelativeMarginProxy, StableMarginMomentum,
)
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEP = "=" * 64

def main():
    print(f"H511: VolScaledCSStarCombo (equal-weight H412+H415+H429+H447 + vol-scaling)\n{SEP}")
    tickers = get_all_historical_tickers()
    panel = load_stocks_panel(tickers)
    membership = load_sp500_members(); common = panel.index.intersection(membership.index)
    panel = panel.loc[common]; membership = membership.loc[common]
    spy = load_research("SPY"); spy_rets = spy["adj_close"].pct_change().fillna(0.0)
    sectors = pd.read_parquet(DATA_DIR / "sp500_sectors.parquet")["sector"]
    pt = panel.columns.tolist()
    gp = load_fundamentals_panel(pt, "gross_profit", panel.index)
    ta = load_fundamentals_panel(pt, "total_assets", panel.index)
    ni = load_fundamentals_panel(pt, "net_income", panel.index)
    rev = load_fundamentals_panel(pt, "revenue", panel.index)
    ltd = load_fundamentals_panel(pt, "long_term_debt", panel.index)
    ocf = load_fundamentals_panel(pt, "operating_cf", panel.index)
    folds = make_folds(panel, train_years=5, test_years=2)
    print(f"Data: {panel.shape[0]} days, {panel.shape[1]} tickers")

    cs_strategies = {
        "H412": lambda tp, tm: FundamentalDivergenceAvoid(net_income=ni, total_assets=ta, sectors=sectors),
        "H415": lambda tp, tm: StableMarginMomentum(gross_profit=gp, revenue=rev, sectors=sectors),
        "H429": lambda tp, tm: SectorRelativeMarginProxy(gross_profit=gp, revenue=rev, sectors=sectors),
        "H447": lambda tp, tm: CreditQualityMomentumCombo(gross_profit=gp, total_assets=ta, long_term_debt=ltd, operating_cf=ocf, sectors=sectors),
    }
    cs_returns = {}
    for name, factory in cs_strategies.items():
        wf = walk_forward_cross_sectional(panel, membership, factory, folds=folds, spy_returns=spy_rets)
        cs_returns[name] = wf.combined_returns
        print(f"  {name}: Sharpe {sharpe(wf.combined_returns):.3f}  MaxDD {max_drawdown(wf.combined_returns):.2%}")

    idx = cs_returns["H412"].index
    for k in cs_returns: cs_returns[k] = cs_returns[k].reindex(idx).fillna(0.0)
    blend = pd.concat(cs_returns.values(), axis=1).mean(axis=1)
    print(f"Equal-weight blend: Sharpe {sharpe(blend):.3f}  MaxDD {max_drawdown(blend):.2%}")

    # Portfolio-level vol-scaling (H086 mechanism: scale * min(1, 8%/realized_vol))
    port_vol = blend.rolling(21, min_periods=10).std().shift(1) * np.sqrt(252)
    scale = (0.08 / port_vol.replace(0, np.nan)).clip(upper=1.0).fillna(1.0)
    vol_scaled = blend * scale
    lo, hi = bootstrap_sharpe_ci(vol_scaled, n_boot=5000)
    s = sharpe(vol_scaled); c = cagr(vol_scaled); dd = max_drawdown(vol_scaled)
    print(f"\nVol-scaled: Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]  CAGR {c:.2%}  MaxDD {dd:.2%}")

    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds):
        mask = (vol_scaled.index >= fold.test_start) & (vol_scaled.index < fold.test_end)
        fr = vol_scaled[mask]; fs = sharpe(fr) if len(fr) > 10 else 0.0
        fold_sharpes.append(float(fs))
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    fs_spy = make_folds(spy, train_years=5, test_years=2)
    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=fs_spy)
    corr = vol_scaled.corr(wh.combined_returns.reindex(vol_scaled.index))
    print(f"\nH001 correlation: {corr:+.4f}"); decision = "KEEP" if s > 0.51 and lo > 0 else "kill"; print(f"Decision: {decision}")
    n_trades = sum(walk_forward_cross_sectional(panel, membership, f, folds=folds, spy_returns=spy_rets).combined_report.n_trades for f in cs_strategies.values())
    write_summary(
        hypothesis_id="511", name="vol-scaled-cs-star-combo", strategy_class="VolScaledCSStarCombo",
        family="Set7 Cross-Program Synergy", sharpe=s, ci_low=lo, ci_high=hi, cagr=c,
        max_drawdown=dd, n_trades=n_trades, h001_corr=float(corr),
        fold_sharpes=fold_sharpes, decision=decision,
        notes="Equal-weight H412+H415+H429+H447 CS star blend with portfolio-level vol-scaling (8% target). H086's vol-scaling applied to CS ensemble. Tests if vol-scaling mechanism is universal across AA and CS portfolios.",
        canonical_params={"blend": "equal_weight", "vol_target": 0.08, "vol_lookback": 21},
    )

if __name__ == "__main__":
    main()
