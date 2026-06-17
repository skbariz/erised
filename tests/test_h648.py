"""H648 DeployableEqualWeight: equal-weight portfolio equals the column mean."""

from __future__ import annotations

import numpy as np


def test_equal_weight_is_column_mean():
    n = 5
    w = np.full(n, 1.0 / n)
    assert abs(w.sum() - 1.0) < 1e-12
    mat = np.arange(20.0).reshape(4, 5)
    port = mat @ w
    assert np.allclose(port, mat.mean(axis=1))
