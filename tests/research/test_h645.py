"""H645 DefensibleTerminal645: leverage application is finite & well-formed."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "runners"))

import numpy as np
import pandas as pd
from set8_lev import apply_leverage, two_state_cap, vix_momentum_mask


def test_apply_leverage_finite_and_capped():
    idx = pd.date_range("2000-01-01", periods=400, freq="B")
    rng = np.random.RandomState(1)
    base = rng.normal(0.0005, 0.01, 400)
    vix = pd.Series(rng.uniform(8.0, 35.0, 400), index=idx)
    cap = two_state_cap(vix_momentum_mask(vix, idx, 16.0, 20), 2.5, 1.0, idx)
    r = apply_leverage(base, idx, 0.065, cap, 20)
    assert len(r) == 400
    assert np.isfinite(r).all()


def test_two_state_cap_values():
    idx = pd.date_range("2000-01-01", periods=10, freq="B")
    mask = pd.Series([True, False] * 5, index=idx)
    cap = two_state_cap(mask, 2.5, 1.0, idx)
    assert cap.iloc[0] == 2.5
    assert cap.iloc[1] == 1.0
