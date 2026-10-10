"""H670 DeployableInvMaxDD670: inverse-MaxDD parity down-weights the deepest-drawdown sleeve."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "runners"))

import numpy as np
from run_h670 import inv_maxdd_weights, standalone_maxdd


def test_inv_maxdd_valid_simplex():
    rng = np.random.RandomState(0)
    w = inv_maxdd_weights(rng.randn(2000, 4) * 0.01)
    assert np.all(w > 0) and np.isclose(w.sum(), 1.0)


def test_inv_maxdd_downweights_deep_drawdown_sleeve():
    rng = np.random.RandomState(3)
    smooth = rng.randn(3000) * 0.003 + 0.0002  # shallow-drawdown sleeve
    crash = rng.randn(3000) * 0.003 + 0.0002
    crash[1000:1120] -= 0.01  # sustained losing run -> deep drawdown
    w = inv_maxdd_weights(np.column_stack([smooth, crash]))
    assert standalone_maxdd(crash) > standalone_maxdd(smooth)
    assert w[0] > w[1]  # the smoother sleeve gets more weight
