"""H668 DeployableMaxSortino668: max-Sortino loads onto the higher return-per-downside sleeve."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "runners"))

import numpy as np
from run_h668 import sortino_weights


def test_sortino_valid_simplex():
    rng = np.random.RandomState(0)
    w = sortino_weights(rng.randn(2000, 3) * 0.01 + 0.0002)
    assert np.all(w > -1e-6) and np.isclose(w.sum(), 1.0)


def test_sortino_prefers_dominant_sleeve():
    rng = np.random.RandomState(2)
    base = rng.randn(3000) * 0.01
    s0 = base + 0.0004  # same risk, strictly higher mean -> dominates on Sortino
    s1 = base
    w = sortino_weights(np.column_stack([s0, s1]))
    assert w[0] > w[1]  # weight concentrates on the dominant sleeve
