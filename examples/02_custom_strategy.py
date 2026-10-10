"""Write your own strategy, check it for lookahead bias, and validate it.

    uv run python examples/02_custom_strategy.py

A strategy subclasses ``Strategy`` and returns a target position in [-1, 1] for
every date. The position decided at the close of day t earns day t+1's return;
the engine applies that one-bar shift for you, so do not shift again inside
``generate_positions``.
"""

from __future__ import annotations

import pandas as pd

from erised.backtest.engine import assert_no_lookahead
from erised.data.synthetic import synthetic_bars
from erised.metrics.performance import bootstrap_sharpe_ci
from erised.strategies import Strategy
from erised.validation.walk_forward import make_folds, walk_forward


class VolTargetedTrend(Strategy):
    """Long above the 200-day average, sized to a target annualized volatility."""

    def __init__(self, ma_days: int = 200, vol_days: int = 20, target_vol: float = 0.10):
        self.ma_days = ma_days
        self.vol_days = vol_days
        self.target_vol = target_vol
        self.name = f"vol_targeted_trend_{ma_days}"

    def generate_positions(self, prices: pd.DataFrame) -> pd.Series:
        px = prices["adj_close"]
        in_trend = (px > px.rolling(self.ma_days, min_periods=self.ma_days).mean()).astype(float)
        realized_vol = px.pct_change().rolling(self.vol_days).std() * (252**0.5)
        size = (self.target_vol / realized_vol).clip(upper=1.0).fillna(0.0)
        return (in_trend * size).rename("position")


def main() -> None:
    prices = synthetic_bars()

    # Raises AssertionError if truncating future data changes any past position.
    assert_no_lookahead(VolTargetedTrend().generate_positions, prices)
    print("Lookahead check passed")

    folds = make_folds(prices, train_years=5, test_years=2)
    wf = walk_forward(prices, lambda _train: VolTargetedTrend(), folds=folds)
    lo, hi = bootstrap_sharpe_ci(wf.combined_returns)
    rep = wf.combined_report

    keep = lo > 0
    print(f"Sharpe {rep.sharpe:.2f}  CI [{lo:.2f}, {hi:.2f}]  MaxDD {rep.max_drawdown:.1%}")
    print("Decision:", "KEEP" if keep else "KILL (confidence interval includes zero)")


if __name__ == "__main__":
    main()
