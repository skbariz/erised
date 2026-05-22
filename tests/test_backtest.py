"""Tests for the backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.engine import assert_no_lookahead, run_backtest
from quant.strategies.base import (
    BuyAndHold,
    MovingAverageCrossover,
)


def _make_prices(returns: list[float], start: str = "2020-01-02") -> pd.DataFrame:
    """Build a tiny prices DataFrame from a list of daily returns."""
    idx = pd.bdate_range(start=start, periods=len(returns) + 1)
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "adj_close": prices,
            "volume": [1_000_000] * len(prices),
        },
        index=idx,
    )
    return df


def test_buy_and_hold_matches_benchmark():
    """A buy-and-hold strategy should equal the benchmark (minus initial trade cost)."""
    prices = _make_prices([0.01, -0.02, 0.03, 0.005, -0.01])
    bh = BuyAndHold()
    positions = bh.generate_positions(prices)
    result = run_backtest(prices, positions, commission_bps=0, slippage_bps=0)
    # With zero costs, BH net returns == bench returns
    np.testing.assert_allclose(result.net_returns.values, result.bench_returns.values, atol=1e-12)


def test_no_lookahead_position_shifted():
    """Position at time t must be applied to period (t, t+1].

    If we set position=+1 on day 0, we should NOT earn the return on day 0;
    we should earn the return on day 1.
    """
    prices = _make_prices([0.05])  # 5% return on day 1
    # Set position to +1 starting day 0
    positions = pd.Series([1.0, 1.0], index=prices.index)
    result = run_backtest(prices, positions, commission_bps=0, slippage_bps=0)
    # Day 0 return should be 0 (we hadn't traded yet)
    # Day 1 return should be 5%
    assert result.net_returns.iloc[0] == 0.0
    assert result.net_returns.iloc[1] == pytest.approx(0.05)


def test_costs_charged_on_position_change():
    """A position flip from 0 to 1 should incur cost = (commission + slippage) * 1.0."""
    prices = _make_prices([0.0, 0.0, 0.0])
    positions = pd.Series([0.0, 1.0, 1.0, 1.0], index=prices.index)
    result = run_backtest(prices, positions, commission_bps=1.0, slippage_bps=1.0)
    # Cost rate = 2 bps = 0.0002 per unit traded
    # First nonzero position appears day 1 -> charge on day 2 (shifted)
    assert result.costs.iloc[0] == 0.0
    assert result.costs.iloc[1] == 0.0
    assert result.costs.iloc[2] == pytest.approx(0.0002)
    assert result.costs.iloc[3] == 0.0


def test_short_position_inverts_returns():
    """Going short with +5% market move should produce -5% gross return."""
    prices = _make_prices([0.05])
    positions = pd.Series([-1.0, -1.0], index=prices.index)
    result = run_backtest(prices, positions, commission_bps=0, slippage_bps=0)
    assert result.net_returns.iloc[1] == pytest.approx(-0.05)


def test_position_magnitude_validated():
    """Positions > 1 or < -1 should raise."""
    prices = _make_prices([0.0])
    positions = pd.Series([2.0, 2.0], index=prices.index)
    with pytest.raises(ValueError, match="position magnitude"):
        run_backtest(prices, positions)


# ───────────────────────────────────────────────────────────────────────────
# H001 – MovingAverageCrossover
# ───────────────────────────────────────────────────────────────────────────

def test_ma_crossover_no_lookahead():
    """The MA crossover strategy must be lookahead-free."""
    # Build a longer series so MAs can warm up
    rng = np.random.default_rng(0)
    n = 1000
    rets = rng.normal(0.0005, 0.01, n).tolist()
    prices = _make_prices(rets)
    strat = MovingAverageCrossover(fast=20, slow=50)
    assert_no_lookahead(strat.generate_positions, prices, n_checks=10)


def test_buy_and_hold_compounds_correctly():
    """A 10% then -5% return: equity = 1.10 * 0.95 = 1.045"""
    prices = _make_prices([0.10, -0.05])
    positions = pd.Series([1.0, 1.0, 1.0], index=prices.index)
    result = run_backtest(prices, positions, commission_bps=0, slippage_bps=0)
    # Day 0: no position yet (shifted) -> return 0, equity 1
    # Day 1: applied position 1, return 10%, equity 1.10
    # Day 2: applied position 1, return -5%, equity 1.045
    assert result.equity.iloc[0] == pytest.approx(1.0)
    assert result.equity.iloc[1] == pytest.approx(1.10)
    assert result.equity.iloc[2] == pytest.approx(1.045)
