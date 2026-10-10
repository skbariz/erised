"""Tests for H069: VixTermStructureProxy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.strategies import VixTermStructureProxy


def _prices(vix_arr: list[float] | None = None, n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2000-01-01", periods=n, freq="B")
    rng = np.random.default_rng(11)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    vix = np.array(vix_arr) if vix_arr is not None else np.full(n, 15.0)
    return pd.DataFrame({"adj_close": spy, "vix_close": vix}, index=idx)


def test_raises_without_vix():
    strat = VixTermStructureProxy()
    df = pd.DataFrame(
        {"adj_close": [100.0] * 50}, index=pd.date_range("2000-01-01", periods=50, freq="B")
    )
    with pytest.raises(ValueError, match="vix_close"):
        strat.generate_positions(df)


def test_returns_series():
    pos = VixTermStructureProxy().generate_positions(_prices())
    assert isinstance(pos, pd.Series)
    assert pos.name == "position"


def test_positions_binary():
    pos = VixTermStructureProxy().generate_positions(_prices())
    assert set(pos.unique()).issubset({0.0, 1.0})


def test_contango_produces_long():
    # VIX history: 20 for warmup, then sudden drop to 10 (ratio 10/20 = 0.5 < 0.85)
    n = 200
    vix = [20.0] * 100 + [10.0] * 100
    pos = VixTermStructureProxy(ma_days=63, contango_thresh=0.85).generate_positions(
        _prices(vix_arr=vix, n=n)
    )
    # After warmup when VIX drops below 85% of its 63d MA, should be long
    assert pos.iloc[110:].sum() > 0


def test_backwardation_produces_flat():
    # VIX history: 10 for warmup, then jump to 30 (ratio 30/10 = 3.0 > 1.15)
    n = 200
    vix = [10.0] * 100 + [30.0] * 100
    pos = VixTermStructureProxy(ma_days=63, backwardation_thresh=1.15).generate_positions(
        _prices(vix_arr=vix, n=n)
    )
    assert pos.iloc[115:].sum() == pytest.approx(0.0)


def test_hysteresis_holds_prior_position():
    # VIX constant = 15, MA = 15, ratio = 1.0 → in neutral band (0.85-1.15) → hold 0
    n = 200
    vix = [15.0] * n
    pos = VixTermStructureProxy(
        ma_days=30, contango_thresh=0.85, backwardation_thresh=1.15
    ).generate_positions(_prices(vix_arr=vix, n=n))
    # Started at 0 and stayed neutral band → should remain 0
    assert pos.iloc[35:].sum() == pytest.approx(0.0)


def test_flat_before_warmup():
    pos = VixTermStructureProxy(ma_days=63).generate_positions(_prices(n=200))
    assert pos.iloc[:64].sum() == pytest.approx(0.0)
