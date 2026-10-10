"""Every script in examples/ runs to completion."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).resolve().parents[2] / "examples").glob("*.py"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_examples_run(path, capsys):
    runpy.run_path(str(path), run_name="__main__")
    assert "Sharpe" in capsys.readouterr().out
