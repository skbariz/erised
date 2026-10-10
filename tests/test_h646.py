"""H646 DeployableCalmar: lagged regime mask is well-formed."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "runners"))

import numpy as np
import pandas as pd
from set8_lev import two_state_cap, vix_momentum_mask


def test_lagged_mask_shape_and_fill():
    idx = pd.date_range("2000-01-01", periods=80, freq="B")
    vix = pd.Series(np.linspace(28.0, 9.0, 80), index=idx)
    lagged = vix_momentum_mask(vix, idx, 16.0, 20).shift(1).fillna(True)
    assert len(lagged) == 80
    # shift(1).fillna(True) yields object dtype, but values are boolean and two_state_cap
    # consumes them via np.where -- so check the values, not the dtype.
    assert set(lagged.unique()) <= {True, False}
    assert bool(lagged.iloc[0]) is True  # first day filled True (no prior signal)


def test_two_state_cap_on_lagged_bull_is_within_bounds():
    idx = pd.date_range("2000-01-01", periods=80, freq="B")
    vix = pd.Series(np.linspace(28.0, 9.0, 80), index=idx)
    bull = vix_momentum_mask(vix, idx, 16.0, 20).shift(1).fillna(True)
    cap = two_state_cap(bull, 2.5, 1.0, idx)
    assert cap.min() >= 1.0 and cap.max() <= 2.5
