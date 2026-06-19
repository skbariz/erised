"""H652 DeployableSparse652: top-K sparsification keeps exactly K sleeves and renormalizes."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
from run_h652 import sparsify


def test_sparsify_keeps_topk_and_renormalizes():
    w = np.array([0.40, 0.05, 0.30, 0.20, 0.05])
    out = sparsify(w, 2)
    assert int((out > 0).sum()) == 2  # exactly K nonzero
    assert np.isclose(out.sum(), 1.0)  # renormalized
    # the two largest (indices 0 and 2) are the survivors
    assert out[0] > 0 and out[2] > 0
    assert out[1] == 0 and out[3] == 0 and out[4] == 0


def test_sparsify_k_geq_n_is_identity_up_to_norm():
    w = np.array([0.5, 0.3, 0.2])
    out = sparsify(w, 5)
    assert np.allclose(out, w)
