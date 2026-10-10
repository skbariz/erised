"""H498: CoreDiversifierSleeve -- 33% H078 (SeasonalRiskParity) + 33% H086 (LowVolTarget) + 33% H473 (LongShortPCAReversal).
Program's 3 lowest H001-corr strategies blended equally. Seasonal + vol-control + market-neutral.
Run: uv run python scripts/run_h498.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from write_summary import write_summary
from erised.data.loader import load_research
from erised.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
from erised.metrics.performance import bootstrap_sharpe_ci, cagr, max_drawdown, sharpe
from erised.strategies.base import LowVolTargetPortfolio, MovingAverageCrossover, SeasonalRiskParity
from erised.strategies.cross_sectional import LongShortPCAReversal
from erised.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional, walk_forward_multi
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SEP = "=" * 64

def main():
    print(f"H498: CoreDiversifierSleeve (33% H078 + 33% H086 + 33% H473)\n{SEP}")
    # AA leg (H078 + H086): SPY + TLT
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    prices_aa = spy.join(tlt, how="left"); prices_aa["tlt_close"] = prices_aa["tlt_close"].ffill()
    folds_aa = make_folds(prices_aa, train_years=5, test_years=2)

    wf_h078 = walk_forward_multi(prices_aa, lambda _: SeasonalRiskParity(), folds=folds_aa)
    wf_h086 = walk_forward_multi(prices_aa, lambda _: LowVolTargetPortfolio(), folds=folds_aa)
    print(f"H078 (SeasonalRiskParity): Sharpe {sharpe(wf_h078.combined_returns):.3f}  MaxDD {max_drawdown(wf_h078.combined_returns):.2%}")
    print(f"H086 (LowVolTarget):       Sharpe {sharpe(wf_h086.combined_returns):.3f}  MaxDD {max_drawdown(wf_h086.combined_returns):.2%}")

    # CS leg (H473): LongShortPCAReversal
    tickers = get_all_historical_tickers()
    panel = load_stocks_panel(tickers)
    membership = load_sp500_members(); common = panel.index.intersection(membership.index)
    panel = panel.loc[common]; membership = membership.loc[common]
    spy_rets = spy["adj_close"].pct_change().fillna(0.0)
    sectors = pd.read_parquet(DATA_DIR / "sp500_sectors.parquet")["sector"]
    folds_cs = make_folds(panel, train_years=5, test_years=2)
    wf_h473 = walk_forward_cross_sectional(
        panel, membership,
        lambda tp, tm: LongShortPCAReversal(train_prices=tp, sectors=sectors),
        folds=folds_cs, spy_returns=spy_rets,
    )
    print(f"H473 (LongShortPCA):       Sharpe {sharpe(wf_h473.combined_returns):.3f}  MaxDD {max_drawdown(wf_h473.combined_returns):.2%}")

    # Blend 33%/33%/33%
    r078 = wf_h078.combined_returns; r086 = wf_h086.combined_returns; r473 = wf_h473.combined_returns
    common_idx = r078.index.intersection(r086.index).intersection(r473.index)
    blended = (r078.loc[common_idx] + r086.loc[common_idx] + r473.loc[common_idx]) / 3.0
    lo, hi = bootstrap_sharpe_ci(blended, n_boot=5000)
    s = sharpe(blended); c = cagr(blended); dd = max_drawdown(blended)
    print(f"\n33/33/33 Blend: Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]  CAGR {c:.2%}  MaxDD {dd:.2%}")

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

    n_trades = wf_h078.combined_report.n_trades + wf_h086.combined_report.n_trades + wf_h473.combined_report.n_trades
    write_summary(
        hypothesis_id="498", name="core-diversifier-sleeve", strategy_class="CoreDiversifierSleeve",
        family="Set7 Continuous Macro AA", sharpe=s, ci_low=lo, ci_high=hi, cagr=c,
        max_drawdown=dd, n_trades=n_trades, h001_corr=float(corr),
        fold_sharpes=fold_sharpes, decision=decision,
        notes="Equal blend of program's 3 lowest H001-corr actively-traded strategies: H078 SeasonalRiskParity (corr +0.008), H086 LowVolTargetPortfolio (corr +0.413), H473 LongShortPCAReversal (corr +0.029). Seasonal + vol-control + market-neutral alpha.",
        canonical_params={"blend": 0.333, "h078_vol_lookback": 60, "h086_vol_target": 0.08, "h473_n_components": 10},
    )

if __name__ == "__main__":
    main()
