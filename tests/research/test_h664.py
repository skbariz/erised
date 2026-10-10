"""H664 DeployableTurnoverPenalized664: the L1 penalty worsens the objective away from the anchor only."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "runners"))

import numpy as np
import pandas as pd
from run_h664 import lev_sharpe_l1_neg


def test_l1_penalty_worsens_objective_away_from_anchor():
    rng = np.random.RandomState(0)
    mat = np.column_stack([0.0006 + 0.01 * rng.randn(400), 0.0004 + 0.01 * rng.randn(400)])
    idx = pd.date_range("2000-01-01", periods=400, freq="B")
    anchor = np.array([0.5, 0.5])
    w_far = np.array([1.0, 0.0])  # L1 distance 1.0 from the anchor
    no_pen = lev_sharpe_l1_neg(mat, idx, 100.0, 1.0, anchor, 0.0)(w_far)
    with_pen = lev_sharpe_l1_neg(mat, idx, 100.0, 1.0, anchor, 0.3)(w_far)
    assert with_pen > no_pen  # penalty raises (worsens) the minimized objective off-anchor


def test_no_penalty_at_the_anchor():
    rng = np.random.RandomState(1)
    mat = np.column_stack([0.0006 + 0.01 * rng.randn(400), 0.0004 + 0.01 * rng.randn(400)])
    idx = pd.date_range("2000-01-01", periods=400, freq="B")
    anchor = np.array([0.5, 0.5])
    a = lev_sharpe_l1_neg(mat, idx, 100.0, 1.0, anchor, 0.0)(anchor)
    b = lev_sharpe_l1_neg(mat, idx, 100.0, 1.0, anchor, 0.3)(anchor)
    assert np.isclose(a, b)  # zero L1 distance at the anchor -> identical objective
