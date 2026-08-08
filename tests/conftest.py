"""Shared fixtures: extract the committed train.json.zip when train.json is absent.

CI does not commit the 51MB extracted file (it is gitignored), so the
real-data tests extract from data/train.json.zip on demand.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

EXTRACT_DIR = Path("/tmp/m2c_train_extract")


@pytest.fixture(scope="session")
def train_rows() -> list[dict]:
    path = Path("data/train.json")
    if not path.exists():
        zpath = Path("data/train.json.zip")
        if not zpath.exists():
            pytest.skip("data/train.json and data/train.json.zip both missing")
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath) as zf:
            zf.extract("train.json", EXTRACT_DIR)
        path = EXTRACT_DIR / "train.json"
    rows: list[dict] = json.loads(path.read_text())
    return rows
