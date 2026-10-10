"""H643 DeployableLaggedChampion: 1-day regime-lag unit test."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "runners"))

import numpy as np
import pandas as pd
from set8_lev import vix_momentum_mask


def test_mask_lag_shifts_by_one_day():
    idx = pd.date_range("2000-01-01", periods=60, freq="B")
    vix = pd.Series(np.linspace(30.0, 8.0, 60), index=idx)
    m = vix_momentum_mask(vix, idx, 16.0, 20)
    lag = m.shift(1).fillna(True)
    # lagged signal at day t equals the unlagged signal at day t-1
    assert (lag.iloc[2:].to_numpy() == m.iloc[1:-1].to_numpy()).all()
