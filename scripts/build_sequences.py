"""Turn each customer's statements into a fixed-size sequence for the sequence models.

    python scripts/build_sequences.py

Unlike build_features.py, nothing is summarised: every statement is kept, in
date order. Each customer becomes

    x     13 x F  numeric features of each statement, plus missing flags
    cat   13 x 11 category codes, for the model's embedding layers
    mask  13      True for a real statement, False for padding

Customers with fewer than 13 statements are padded at the end, so position 0 is
always the customer's first statement, and `mask` tells the model which
positions are real. The audit found that history length predicts default, so
it must stay visible rather than being filled in.

Preprocessing, all of it fitted on the training customers' statements only:

  - -1 in integer columns becomes missing, as everywhere else in the project.
  - Each numeric feature is standardised with the training mean and standard
    deviation, then clipped to [-5, 5]. A few features have extreme outliers,
    which would otherwise dominate the gradients of a neural network. Trees do
    not care about scale; neural networks do.
  - Missing values become 0 (the training mean), and every feature that has
    any missing values in training gets a 0/1 flag, so the model can tell
    "missing" apart from "exactly average". The audit showed missing values are
    common and possibly informative.
  - Category codes are shifted up by one, so -1 (missing) becomes 0 and is
    learned as a category of its own. Padding also uses 0 but is masked out.

Arrays are stored as float16, as the Project Definition planned, to halve the
memory. Writes data/sequences/<split>_{x,cat,mask,y}.npy plus meta.json with the
feature names and the fitted normalisation.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from audit_data import CATEGORICAL, DATE, ID, MISSING_CODE, TARGET
from make_split import SPLIT_FILE
from paths import DATA, TRAIN_PARQUET

SEQ = DATA / "sequences"
MAX_LEN = 13
CLIP = 5.0


def main() -> None:
    t0 = time.time()
    st = pd.read_parquet(TRAIN_PARQUET)
    st[DATE] = pd.to_datetime(st[DATE])
    st = st.sort_values([ID, DATE], kind="stable").reset_index(drop=True)

    split = pd.read_parquet(SPLIT_FILE).set_index(ID)
    codes, ids = pd.factorize(st[ID])
    pos = st.groupby(codes).cumcount().to_numpy()
    assert pos.max() < MAX_LEN
    in_train = (st[ID].map(split["split"]) == "train").to_numpy()
    n = len(ids)
    print(f"Loaded {len(st):,} statements for {n:,} customers ({time.time() - t0:.0f}s).")

    numeric = [c for c in st.columns if c not in (ID, DATE, *CATEGORICAL)]

    def values(c: str) -> np.ndarray:
        v = st[c].to_numpy(np.float32)
        if pd.api.types.is_integer_dtype(st[c]):
            v[st[c].to_numpy() == MISSING_CODE] = np.nan
        return v

    # First pass: which features have missing values among training statements.
    # Knowing the final width up front lets the array be allocated once.
    flagged = [c for c in numeric if np.isnan(values(c)[in_train]).any()]
    flag_pos = {c: len(numeric) + k for k, c in enumerate(flagged)}
    x = np.zeros((n, MAX_LEN, len(numeric) + len(flagged)), dtype=np.float16)

    mean, std = {}, {}
    for j, c in enumerate(numeric):
        v = values(c)
        missing = np.isnan(v)
        m, s = float(np.nanmean(v[in_train])), float(np.nanstd(v[in_train]))
        s = s if s > 0 else 1.0
        mean[c], std[c] = m, s
        z = np.clip((v - m) / s, -CLIP, CLIP)
        z[missing] = 0.0
        x[codes, pos, j] = z
        if c in flag_pos:
            x[codes, pos, flag_pos[c]] = missing
    print(f"  normalised {len(numeric)} numeric features; {len(flagged)} get a missing flag "
          f"({time.time() - t0:.0f}s)")

    cat = np.zeros((n, MAX_LEN, len(CATEGORICAL)), dtype=np.int8)
    cat[codes, pos] = st[CATEGORICAL].to_numpy() + 1
    cardinality = [int(cat[..., k].max()) + 1 for k in range(len(CATEGORICAL))]

    mask = np.zeros((n, MAX_LEN), dtype=bool)
    mask[codes, pos] = True

    del st
    labels = split.loc[ids]
    y = labels[TARGET].to_numpy(np.int8)
    which = labels["split"].to_numpy()

    SEQ.mkdir(parents=True, exist_ok=True)
    for name in ["train", "valid", "test"]:
        rows = np.flatnonzero(which == name)
        np.save(SEQ / f"{name}_x.npy", x[rows])
        np.save(SEQ / f"{name}_cat.npy", cat[rows])
        np.save(SEQ / f"{name}_mask.npy", mask[rows])
        np.save(SEQ / f"{name}_y.npy", y[rows])
        pd.Series(ids[rows], name=ID).to_frame().to_parquet(SEQ / f"{name}_ids.parquet", index=False)
        print(f"  {name}: {len(rows):,} customers, x {(len(rows), *x.shape[1:])}, "
              f"{100 * mask[rows].all(axis=1).mean():.1f}% with all {MAX_LEN} months")

    meta = {
        "numeric": numeric, "flagged": flagged, "categorical": CATEGORICAL,
        "cardinality": cardinality, "max_len": MAX_LEN, "clip": CLIP,
        "mean": mean, "std": std,
    }
    (SEQ / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"\n{len(numeric)} numeric + {len(flagged)} missing flags = {x.shape[2]} inputs per month, "
          f"plus {len(CATEGORICAL)} categorical codes.")
    print(f"Written to {SEQ} in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
