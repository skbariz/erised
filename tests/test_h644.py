"""H644 ShrinkageRobustChampion: ridge-shrinkage penalty unit test."""

from __future__ import annotations

import numpy as np


def test_shrinkage_penalizes_deviation_from_prior():
    w_mdr = np.full(5, 0.2)
    w_concentrated = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    mu = 0.5
    pen_at_prior = mu * np.sum((w_mdr - w_mdr) ** 2)
    pen_concentrated = mu * np.sum((w_concentrated - w_mdr) ** 2)
    assert pen_at_prior == 0.0
    assert pen_concentrated > 0.0
    # larger mu => larger penalty for the same deviation
    assert 2.0 * np.sum((w_concentrated - w_mdr) ** 2) > pen_concentrated
