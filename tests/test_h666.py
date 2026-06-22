"""H666 DeployableHRP666: Hierarchical Risk Parity splits weight across the cluster tree."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
from run_h666 import hrp_weights


def _two_block_matrix(seed=0):
    rng = np.random.RandomState(seed)
    fa = rng.randn(3000) * 0.01  # block-A common factor
    fb = rng.randn(3000) * 0.01  # block-B common factor (independent of A)
    cols = [fa + rng.randn(3000) * 0.002 for _ in range(3)]
    cols += [fb + rng.randn(3000) * 0.002 for _ in range(3)]
    return np.column_stack(cols)


def test_hrp_valid_simplex():
    w = hrp_weights(_two_block_matrix())
    assert np.all(w > 0) and np.isclose(w.sum(), 1.0)


def test_hrp_splits_evenly_across_symmetric_blocks():
    w = hrp_weights(_two_block_matrix())
    block_a, block_b = w[:3].sum(), w[3:].sum()
    # two symmetric, mutually-independent blocks should each receive ~half the book
    assert abs(block_a - block_b) < 0.20
