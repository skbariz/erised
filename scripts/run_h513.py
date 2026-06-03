"""H513: QualitySeasonalMarketNeutral -- Nov-Apr: H021; May-Oct: 50% TLT + 50% H473 (L/S reversal).
H078's structure: replace May-Oct 100% TLT with 50% TLT + 50% market-neutral L/S alpha.
Run: uv run python scripts/run_h513.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from write_summary import write_summary
from quant.data.loader import load_research
from quant.data.stocks import get_all_historical_tickers, load_sp500_members, load_stocks_panel
from quant.metrics.performance import bootstrap_sharpe_ci, cagr, max_drawdown, sharpe
from quant.strategies.base import MovingAverageCrossover, RiskParity
from quant.strategies.cross_sectional import LongShortPCAReversal
from quant.validation.walk_forward import make_folds, walk_forward, walk_forward_cross_sectional, walk_forward_multi
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEP = "=" * 64

def main():
    print(f"H513: QualitySeasonalMarketNeutral (Nov-Apr: H021; May-Oct: 50% TLT + 50% H473)\n{SEP}")
    # AA leg: H021
    spy = load_research("SPY")
    tlt = load_research("TLT")[["adj_close"]].rename(columns={"adj_close": "tlt_close"})
    tlt_rets = load_research("TLT")[["adj_close"]]["adj_close"].pct_change()
    prices_aa = spy.join(tlt, how="left"); prices_aa["tlt_close"] = prices_aa["tlt_close"].ffill()
    folds_aa = make_folds(prices_aa, train_years=5, test_years=2)
    wf_h021 = walk_forward_multi(prices_aa, lambda _: RiskParity(60), folds=folds_aa)
    r_h021 = wf_h021.combined_returns
    print(f"H021 (RiskParity): Sharpe {sharpe(r_h021):.3f}  MaxDD {max_drawdown(r_h021):.2%}")

    # CS L/S leg: H473
    import pandas as pd
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
    r_h473 = wf_h473.combined_returns
    print(f"H473 (LongShortPCA): Sharpe {sharpe(r_h473):.3f}  MaxDD {max_drawdown(r_h473):.2%}")

    # Seasonal blend
    common_idx = r_h021.index.intersection(r_h473.index)
    r021 = r_h021.reindex(common_idx); r473 = r_h473.reindex(common_idx).fillna(0.0)
    tlt_r = tlt_rets.reindex(common_idx).fillna(0.0)
    in_season = pd.Series(common_idx.month.isin([11, 12, 1, 2, 3, 4]), index=common_idx)
    r_may_oct = 0.5 * tlt_r + 0.5 * r473
    blended = r021 * in_season.astype(float) + r_may_oct * (~in_season).astype(float)

    lo, hi = bootstrap_sharpe_ci(blended, n_boot=5000)
    s = sharpe(blended); c = cagr(blended); dd = max_drawdown(blended)
    print(f"\nSeasonal blend: Sharpe {s:.3f}  CI [{lo:.3f}, {hi:.3f}]  CAGR {c:.2%}  MaxDD {dd:.2%}")
    print("\nPer-fold Sharpe:")
    fold_sharpes = []
    for i, fold in enumerate(folds_aa):
        mask = (blended.index >= fold.test_start) & (blended.index < fold.test_end)
        fr = blended[mask]; fs = sharpe(fr) if len(fr) > 10 else 0.0
        fold_sharpes.append(float(fs))
        print(f"  Fold {i:>2}: {fs:+.3f}  ({fold.test_start.date()} -- {fold.test_end.date()})")

    fs_spy = make_folds(spy, train_years=5, test_years=2)
    wh = walk_forward(spy, lambda _: MovingAverageCrossover(50, 200), folds=fs_spy)
    corr = blended.corr(wh.combined_returns.reindex(blended.index))
    print(f"\nH001 correlation: {corr:+.4f}"); decision = "KEEP" if s > 0.51 and lo > 0 else "kill"; print(f"Decision: {decision}")
    n_trades = wf_h021.combined_report.n_trades + wf_h473.combined_report.n_trades
    write_summary(
        hypothesis_id="513", name="quality-seasonal-market-neutral", strategy_class="QualitySeasonalMarketNeutral",
        family="Set7 Cross-Program Synergy", sharpe=s, ci_low=lo, ci_high=hi, cagr=c,
        max_drawdown=dd, n_trades=n_trades, h001_corr=float(corr),
        fold_sharpes=fold_sharpes, decision=decision,
        notes="H078's seasonal structure: Nov-Apr = H021 (risk parity); May-Oct = 50% TLT + 50% H473 (LongShortPCAReversal, dollar-neutral). Adds market-neutral L/S alpha to H078's defensive season without adding equity beta.",
        canonical_params={"favorable_months": [11, 12, 1, 2, 3, 4], "may_oct_tlt": 0.50, "may_oct_ls": 0.50},
    )

if __name__ == "__main__":
    main()
