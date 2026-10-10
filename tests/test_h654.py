"""H654 DeployableShrinkage654: the ridge penalty vanishes at equal weight and grows away from it."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "runners"))

import numpy as np
import pandas as pd
from run_h654 import lev_sharpe_ridge_neg


def _mat(periods=300):
    rng = np.random.RandomState(0)
    return np.column_stack([0.001 + 0.01 * rng.randn(periods), 0.001 + 0.01 * rng.randn(periods)])


def test_ridge_penalty_zero_at_equal_weight():
    mat = _mat()
    idx = pd.date_range("2000-01-01", periods=mat.shape[0], freq="B")
    eq = np.full(2, 0.5)
    base = lev_sharpe_ridge_neg(mat, idx, 0.065, 2.5, 0.0, eq)
    shrunk = lev_sharpe_ridge_neg(mat, idx, 0.065, 2.5, 50.0, eq)
    # at w == eq the penalty term is exactly 0, so both objectives coincide
    assert np.isclose(base(eq), shrunk(eq))


def test_ridge_penalty_grows_away_from_equal_weight():
    mat = _mat()
    idx = pd.date_range("2000-01-01", periods=mat.shape[0], freq="B")
    eq = np.full(2, 0.5)
    base = lev_sharpe_ridge_neg(mat, idx, 0.065, 2.5, 0.0, eq)
    shrunk = lev_sharpe_ridge_neg(mat, idx, 0.065, 2.5, 50.0, eq)
    skew = np.array([0.9, 0.1])
    # away from equal weight, the lambda>0 objective is penalized (larger) vs lambda=0
    assert shrunk(skew) > base(skew)
