"""H651 DeployableSortino651: the leveraged-Sortino objective is well-formed."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "runners"))

import numpy as np
import pandas as pd
from run_h651 import lev_sortino_neg


def test_lev_sortino_objective_sign_and_finite():
    idx = pd.date_range("2000-01-01", periods=300, freq="B")
    rng = np.random.RandomState(0)
    # drift (0.002/day) >> sampling error of the mean (0.008/sqrt(300)) -> robustly positive
    mat = np.column_stack(
        [
            0.002 + 0.008 * rng.randn(300),  # positive-drift noisy sleeve
            0.002 + 0.008 * rng.randn(300),
        ]
    )
    neg = lev_sortino_neg(mat, idx, 0.065, 2.5)
    val = neg(np.array([0.5, 0.5]))
    # positive mean return -> maximizing Sortino means the (negated) objective is finite and < 0
    assert np.isfinite(val)
    assert val < 0.0


def test_lev_sortino_no_downside_returns_zero():
    idx = pd.date_range("2000-01-01", periods=60, freq="B")
    mat = np.full((60, 1), 0.001)  # strictly positive: no downside days
    neg = lev_sortino_neg(mat, idx, 0.065, 2.5)
    assert neg(np.array([1.0])) == 0.0
