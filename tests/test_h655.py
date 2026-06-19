"""H655 DeployableCagrDDfloor655: the drawdown-floor penalty fires only when the floor is breached."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pandas as pd
from run_h655 import lev_cagr_ddfloor_neg


def test_penalty_increases_objective_on_deep_drawdown():
    # series that rallies then crashes hard -> drawdown far below the -10% floor
    rets = np.r_[np.full(120, 0.004), np.full(40, -0.02), np.full(120, 0.001)]
    mat = rets.reshape(-1, 1)
    idx = pd.date_range("2000-01-01", periods=len(rets), freq="B")
    # cap=1.0, tv huge -> leverage pinned at ~1.0 so the raw drawdown passes through
    no_pen = lev_cagr_ddfloor_neg(mat, idx, 100.0, 1.0, floor=-0.10, pen=0.0)
    with_pen = lev_cagr_ddfloor_neg(mat, idx, 100.0, 1.0, floor=-0.10, pen=50.0)
    w = np.array([1.0])
    assert with_pen(w) > no_pen(w)  # penalty makes the (minimized) objective worse


def test_no_penalty_when_floor_respected():
    # gentle, monotone-ish series with a shallow drawdown well inside the floor
    rng = np.random.RandomState(0)
    rets = 0.0008 + 0.003 * rng.randn(400)
    mat = rets.reshape(-1, 1)
    idx = pd.date_range("2000-01-01", periods=len(rets), freq="B")
    no_pen = lev_cagr_ddfloor_neg(mat, idx, 100.0, 1.0, floor=-0.50, pen=0.0)
    with_pen = lev_cagr_ddfloor_neg(mat, idx, 100.0, 1.0, floor=-0.50, pen=50.0)
    w = np.array([1.0])
    assert np.isclose(no_pen(w), with_pen(w))  # drawdown above floor -> identical objectives
