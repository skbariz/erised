"""H665 DeployableEnsembleCapstone665: blending champion allocations stays on the simplex."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "runners"))

import numpy as np
from run_h665 import blend_weights


def test_blend_is_convex_combination():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    c = np.array([0.0, 0.0, 1.0])
    w = blend_weights([a, b, c])
    assert np.allclose(w, [1 / 3, 1 / 3, 1 / 3]) and np.isclose(w.sum(), 1.0)


def test_blend_preserves_simplex_for_random_components():
    rng = np.random.RandomState(0)
    ws = [rng.dirichlet(np.ones(8)) for _ in range(6)]
    w = blend_weights(ws)
    assert np.all(w >= 0) and np.isclose(w.sum(), 1.0)
    # the blend is no more concentrated than the most concentrated component
    assert w.max() <= max(c.max() for c in ws) + 1e-12
