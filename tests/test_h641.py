"""H641 DefensibleBlend: VIX-momentum mask + stream-blend unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "runners"))

import numpy as np
import pandas as pd
from set8_lev import vix_momentum_mask


def test_vix_momentum_mask_low_and_not_rising():
    idx = pd.date_range("2000-01-01", periods=100, freq="B")
    vix = pd.Series(np.linspace(30.0, 10.0, 100), index=idx)  # steadily falling VIX
    m = vix_momentum_mask(vix, idx, 16.0, 20)
    assert m.dtype == bool
    assert m.iloc[-1]  # late: VIX low (<16) and falling -> bull
    assert not m.iloc[5]  # early: VIX ~28 (>16) -> bear


def test_blend_is_equal_weight_mean():
    a = np.array([0.01, -0.02, 0.03])
    b = np.array([0.03, 0.00, -0.01])
    blend = np.mean([a, b], axis=0)
    assert np.allclose(blend, [0.02, -0.01, 0.01])
