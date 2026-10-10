"""H662 DeployableSubperiodRobust662: the objective rewards balanced half-period Sharpes."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "runners"))

import numpy as np
import pandas as pd
from run_h662 import _half_sharpes, lev_subperiod_neg


def test_objective_rewards_balanced_halves():
    idx = pd.date_range("2000-01-01", periods=400, freq="B")
    rng = np.random.RandomState(0)
    steady = 0.0008 + 0.005 * rng.randn(400)
    lop = steady.copy()
    lop[200:] -= 0.0016  # second half much weaker -> lower worse-half Sharpe
    w = np.array([1.0])
    neg_steady = lev_subperiod_neg(steady.reshape(-1, 1), idx, 100.0, 1.0, 200)(w)
    neg_lop = lev_subperiod_neg(lop.reshape(-1, 1), idx, 100.0, 1.0, 200)(w)
    assert neg_lop > neg_steady  # weaker second half -> lower min -> worse (higher) objective


def test_half_sharpes_split():
    r = np.r_[np.full(100, 0.002), np.full(100, -0.002)]
    a, b = _half_sharpes(r, 100)
    assert a > 0 > b  # first half positive, second half negative
