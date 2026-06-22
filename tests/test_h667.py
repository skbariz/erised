"""H667 DeployableMinCorr667: the minimum-correlation tilt up-weights the independent sleeve."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
from run_h667 import mincorr_weights


def test_mincorr_valid_simplex():
    rng = np.random.RandomState(0)
    w = mincorr_weights(rng.randn(2000, 4) * 0.01)
    assert np.all(w > 0) and np.isclose(w.sum(), 1.0)


def test_mincorr_overweights_low_correlation_sleeve():
    rng = np.random.RandomState(1)
    common = rng.randn(3000) * 0.01
    s0 = common + rng.randn(3000) * 0.001  # two near-duplicate, highly-correlated sleeves
    s1 = common + rng.randn(3000) * 0.001
    s2 = rng.randn(3000) * 0.01  # independent sleeve, same vol
    w = mincorr_weights(np.column_stack([s0, s1, s2]))
    assert w[2] > w[0] and w[2] > w[1]  # the diversifier gets the largest weight
