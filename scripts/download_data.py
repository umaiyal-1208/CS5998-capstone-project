"""Download the data this project uses from Kaggle.

    python scripts/download_data.py            # download anything missing
    python scripts/download_data.py --force    # delete and re-download
    python scripts/download_data.py --report   # just report what is on disk

Two files are needed:

  train.parquet     The monthly statements, from raddar's integer-dtype Parquet
                    copy of the competition data (about 5.5 million rows). The
                    original CSV is about 16 GB because of injected random noise
                    stored as floats; raddar removed that noise and cast the
                    columns back to integers where possible.
  train_labels.csv  One row per customer with the default target, taken directly
                    from the competition so the labels come from the original
                    source rather than a re-upload.

The competition's test files are not downloaded. Their labels were never
released, so they cannot be used for evaluation here.

Licence position, recorded because the report has to state it accurately:
  The competition data may only be used for non-commercial purposes and may not
  be redistributed (Kaggle competition rules). raddar's dataset is a derived
  copy of that data and inherits the same restriction. Nothing under data/ is
  committed.

Needs a Kaggle API token in ~/.kaggle/kaggle.json (Windows:
C:\\Users\\<you>\\.kaggle\\kaggle.json), and the competition rules accepted at
https://www.kaggle.com/competitions/amex-default-prediction/rules using the same
account. Without the second step the label download fails with 403 Forbidden.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from paths import RAW, REPO_ROOT, TRAIN_LABELS, TRAIN_PARQUET

COMPETITION = "amex-default-prediction"
PARQUET_DATASET = "raddar/amex-data-integer-dtypes-parquet-format"


def kaggle_api():
    """Authenticate lazily, so --report works without a token.

    Importing the kaggle package authenticates as a side effect, so a missing
    token surfaces as an OSError at import time rather than from authenticate().
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except ImportError:
        sys.exit("the kaggle package is missing; run pip install -r requirements.txt")
    except OSError as exc:
        sys.exit(f"Kaggle authentication failed: {exc}\nSee the top of this script for setup.")
    return api


def unzip_if_needed(target: Path) -> None:
    """Kaggle sometimes serves a single file as <name>.zip. Extract it in place."""
    zipped = target.with_name(target.name + ".zip")
    if zipped.exists():
        print(f"  extracting {zipped.name}")
        with zipfile.ZipFile(zipped) as zf:
            zf.extractall(target.parent)
        zipped.unlink()


def download_parquet(api) -> None:
    print(f"\n[train.parquet] from {PARQUET_DATASET}")
    api.dataset_download_file(PARQUET_DATASET, TRAIN_PARQUET.name, path=str(RAW), quiet=False)
    unzip_if_needed(TRAIN_PARQUET)


def download_labels(api) -> None:
    print(f"\n[train_labels.csv] from competition {COMPETITION}")
    api.competition_download_file(COMPETITION, TRAIN_LABELS.name, path=str(RAW), quiet=False)
    unzip_if_needed(TRAIN_LABELS)


def size_mb(path: Path) -> int:
    return round(path.stat().st_size / 1024 / 1024) if path.exists() else 0


def report() -> None:
    print(f"\n{'file':<20}{'present':>9}{'size MB':>10}  path")
    for path in (TRAIN_PARQUET, TRAIN_LABELS):
        try:
            shown = path.relative_to(REPO_ROOT)
        except ValueError:  # AMEX_DATA_DIR points outside the repository
            shown = path
        print(f"{path.name:<20}{'yes' if path.exists() else 'no':>9}{size_mb(path):>10}  {shown}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true", help="delete and re-download both files")
    parser.add_argument("--report", action="store_true", help="report what is on disk and exit")
    args = parser.parse_args()

    if args.report:
        report()
        return

    RAW.mkdir(parents=True, exist_ok=True)
    steps = [(TRAIN_PARQUET, download_parquet), (TRAIN_LABELS, download_labels)]

    api = None
    for path, download in steps:
        if args.force and path.exists():
            print(f"[{path.name}] --force: removing it")
            path.unlink()
        if path.exists():
            print(f"[{path.name}] already present, skipping")
            continue
        api = api or kaggle_api()
        download(api)
        if not path.exists():
            sys.exit(f"[{path.name}] download finished but the file is not at {path}")

    report()


if __name__ == "__main__":
    main()
