"""Runtime configuration.

``ERISED_DATA_DIR`` sets where Parquet market data is read from and written to.
When unset, a source checkout uses ``<repo>/data`` and an installed package uses
``./data`` relative to the current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_ENV = "ERISED_DATA_DIR"


def _default_data_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "pyproject.toml").exists():
        return repo_root / "data"
    return Path.cwd() / "data"


DATA_DIR = (
    Path(os.environ[DATA_DIR_ENV]).expanduser()
    if os.environ.get(DATA_DIR_ENV)
    else _default_data_dir()
)
