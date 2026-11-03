"""Where the data lives. Shared by every script so the location is set in one place.

The data directory defaults to data/ inside the repository. Set AMEX_DATA_DIR to
put it somewhere else, for example outside a OneDrive-synced folder.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("AMEX_DATA_DIR", REPO_ROOT / "data")).expanduser()
RAW = DATA / "raw"

TRAIN_PARQUET = RAW / "train.parquet"
TRAIN_LABELS = RAW / "train_labels.csv"
