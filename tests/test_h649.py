"""H649 DeployableMultiObj: Sharpe x Calmar product objective ordering."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
from set8_lev import portfolio_metrics


def test_product_objective_rewards_both_axes():
    rng = np.random.RandomState(2)
    good = rng.normal(0.0006, 0.008, 2000)  # higher mean, lower vol
    weak = rng.normal(0.0002, 0.012, 2000)
    sh_g, cal_g = portfolio_metrics(good)[:2]
    sh_w, cal_w = portfolio_metrics(weak)[:2]
    # the stronger return stream should score higher on the product objective
    assert sh_g * cal_g > sh_w * cal_w
