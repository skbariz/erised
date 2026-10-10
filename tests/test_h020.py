"""Focused tests for H020: ThreeAssetMomentum."""

from __future__ import annotations

import pandas as pd
import pytest

from erised.backtest.engine import assert_no_lookahead_multi
from erised.strategies.base import ThreeAssetMomentum


def _prices(spy_rets, tlt_rets, gld_rets, start="2010-01-04"):
    n = len(spy_rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    spy = [100.0]
    tlt = [100.0]
    gld = [100.0]
    for r in spy_rets:
        spy.append(spy[-1] * (1 + r))
    for r in tlt_rets:
        tlt.append(tlt[-1] * (1 + r))
    for r in gld_rets:
        gld.append(gld[-1] * (1 + r))
    return pd.DataFrame({"adj_close": spy, "tlt_close": tlt, "gld_close": gld}, index=idx)


def test_no_lookahead():
    prices = _prices([0.001] * 700, [0.0005] * 700, [0.0008] * 700)
    assert_no_lookahead_multi(ThreeAssetMomentum().generate_weights, prices)


def test_picks_best_asset():
    """Asset with highest 12m return gets weight 1.0."""
    prices = _prices([0.003] * 300, [0.001] * 300, [0.002] * 300)
    w = ThreeAssetMomentum().generate_weights(prices)
    assert w["SPY"].iloc[-1] == pytest.approx(1.0)
    assert w["TLT"].iloc[-1] == pytest.approx(0.0)
    assert w["GLD"].iloc[-1] == pytest.approx(0.0)


def test_cash_when_all_negative():
    """All-negative momentum → cash (all weights zero)."""
    prices = _prices([-0.001] * 300, [-0.001] * 300, [-0.001] * 300)
    w = ThreeAssetMomentum().generate_weights(prices)
    assert w.iloc[-1].sum() == pytest.approx(0.0)


def test_at_most_one_asset_active():
    """Only one asset should be non-zero at any given time."""
    prices = _prices(
        [0.001 * (1 + 0.1 * (i % 5)) for i in range(300)],
        [0.0005 * (1 + 0.1 * ((i + 2) % 5)) for i in range(300)],
        [0.0008 * (1 + 0.1 * ((i + 4) % 5)) for i in range(300)],
    )
    w = ThreeAssetMomentum().generate_weights(prices)
    active_count = (w > 0).sum(axis=1)
    assert (active_count <= 1).all()


def test_weight_is_zero_or_one():
    prices = _prices([0.001] * 300, [0.002] * 300, [0.003] * 300)
    w = ThreeAssetMomentum().generate_weights(prices)
    unique_vals = set(w.values.ravel().round(6))
    assert unique_vals.issubset({0.0, 1.0})


def test_gld_wins_when_best_momentum():
    """GLD with highest 12m return → GLD gets weight 1.0."""
    prices = _prices([0.001] * 300, [0.0005] * 300, [0.004] * 300)
    w = ThreeAssetMomentum().generate_weights(prices)
    assert w["GLD"].iloc[-1] == pytest.approx(1.0)


def test_columns_are_correct():
    prices = _prices([0.001] * 300, [0.001] * 300, [0.001] * 300)
    w = ThreeAssetMomentum().generate_weights(prices)
    assert list(w.columns) == ["SPY", "TLT", "GLD"]
