"""H669 DeployableRidge669: raising lambda shrinks the optimized weights toward equal weight."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "runners"))

import numpy as np
from run_h669 import ridge_weights


def _matrix(seed=0):
    rng = np.random.RandomState(seed)
    # one clearly superior sleeve so the unregularized optimum is far from equal weight
    s0 = rng.randn(3000) * 0.01 + 0.0006
    s1 = rng.randn(3000) * 0.01
    s2 = rng.randn(3000) * 0.01
    return np.column_stack([s0, s1, s2])


def _dist_to_equal(w):
    eq = np.full(len(w), 1.0 / len(w))
    return float(np.sqrt(np.sum((w - eq) ** 2)))


def test_ridge_valid_simplex():
    w = ridge_weights(_matrix(), 0.5)
    assert np.all(w > -1e-6) and np.isclose(w.sum(), 1.0)


def test_ridge_shrinks_toward_equal_weight():
    mat = _matrix()
    d0 = _dist_to_equal(ridge_weights(mat, 0.0))
    d_mid = _dist_to_equal(ridge_weights(mat, 5.0))
    d_big = _dist_to_equal(ridge_weights(mat, 20.0))
    assert d_big < d_mid < d0  # stronger penalty monotonically pulls weights toward 1/n
    assert d_big < 0.08  # heavy shrinkage lands very close to equal weight
