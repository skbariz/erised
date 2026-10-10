"""H642 VIXMomAsymCalmar: Calmar objective math unit test."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "runners"))

import numpy as np
from set8_lev import portfolio_metrics


def test_calmar_equals_cagr_over_abs_maxdd():
    rng = np.random.RandomState(0)
    r = rng.normal(0.0004, 0.01, 2000)
    sh, cal, cg, dd, so = portfolio_metrics(r)
    assert dd < 0
    assert abs(cal - cg / abs(dd)) < 1e-9
    assert sh != 0.0
