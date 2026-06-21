"""H663 DeployableHighBorrow663: a wider borrow spread reduces the leveraged Sharpe."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import pandas as pd
from set8_lev import apply_leverage


def test_higher_borrow_reduces_leveraged_return():
    rng = np.random.RandomState(0)
    r = 0.0005 + 0.008 * rng.randn(600)
    idx = pd.date_range("2000-01-01", periods=600, freq="B")
    # tv=0.20 with realized vol ~0.13 -> leverage ~1.5x (>1), so the borrow spread bites
    cheap = apply_leverage(r, idx, 0.20, 3.0, 20, borrow_annual=0.0005)
    dear = apply_leverage(r, idx, 0.20, 3.0, 20, borrow_annual=0.02)
    assert dear.mean() < cheap.mean()  # 200bps borrow drags the leveraged mean below 5bps borrow


def test_no_borrow_drag_when_unlevered():
    rng = np.random.RandomState(1)
    r = 0.0005 + 0.01 * rng.randn(400)
    idx = pd.date_range("2000-01-01", periods=400, freq="B")
    # cap=1.0 pins leverage at <=1x -> no borrow charged regardless of spread
    a = apply_leverage(r, idx, 100.0, 1.0, 20, borrow_annual=0.0005)
    b = apply_leverage(r, idx, 100.0, 1.0, 20, borrow_annual=0.02)
    assert np.allclose(a, b)
