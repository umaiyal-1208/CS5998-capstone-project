"""Summarise each customer's statement history into one row of features.

    python scripts/build_features.py

This is the input for the baseline models, which cannot read a sequence. For
every numeric feature it computes, over the customer's statements:

    last    the value on the latest statement (NaN if missing there)
    first   the value on the earliest statement, used only for `change`
    change  last minus first: the direction the customer is moving in
    mean, std, min, max

For each of the 11 categorical features it keeps the last value and the number
of distinct values seen. It also adds `n_statements`, because the audit found
that customers with shorter histories default more often.

Before aggregating, -1 in the integer columns is turned into NaN. The audit
showed -1 is the missing code there; left in, it would be averaged as if it
were a real value. Categorical columns keep -1 as a category of its own,
because for them "missing" is just another value.

None of this uses the target, and every statistic is computed within one
customer, so building the table on all customers at once leaks nothing between
the splits. Steps that do learn from data (imputation, scaling) are fitted on
the training customers only, in train_baselines.py.

Writes data/features/aggregates.parquet: one row per customer, with the target
and split columns from data/splits/split.parquet.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from audit_data import CATEGORICAL, DATE, ID, MISSING_CODE
from make_split import SPLIT_FILE
from paths import DATA, TRAIN_PARQUET

FEATURES = DATA / "features"
AGG_FILE = FEATURES / "aggregates.parquet"

STATS = ["mean", "std", "min", "max"]
CHUNK = 30  # columns aggregated at a time, to keep memory down


def load() -> tuple[pd.DataFrame, np.ndarray]:
    st = pd.read_parquet(TRAIN_PARQUET)
    # Rows are already grouped by customer and in date order (checked in the
    # audit), but sort anyway so first/last below cannot silently go wrong.
    st[DATE] = pd.to_datetime(st[DATE])
    st = st.sort_values([ID, DATE], kind="stable").reset_index(drop=True)
    codes, _ = pd.factorize(st[ID])
    return st, codes


def numeric_block(st: pd.DataFrame, cols: list[str], codes: np.ndarray,
                  first_idx: np.ndarray, last_idx: np.ndarray) -> pd.DataFrame:
    x = st[cols].astype("float32")
    for c in cols:
        if pd.api.types.is_integer_dtype(st[c]):
            x[c] = x[c].mask(st[c] == MISSING_CODE)

    stats = x.groupby(codes, sort=True).agg(STATS)
    stats.columns = [f"{c}_{s}" for c, s in stats.columns]

    first = x.iloc[first_idx].reset_index(drop=True)
    last = x.iloc[last_idx].reset_index(drop=True)
    change = (last - first).add_suffix("_change")
    return pd.concat([last.add_suffix("_last"), change, stats.reset_index(drop=True)], axis=1)


def categorical_block(st: pd.DataFrame, codes: np.ndarray, last_idx: np.ndarray) -> pd.DataFrame:
    last = st[CATEGORICAL].iloc[last_idx].reset_index(drop=True).add_suffix("_last")
    nunique = st[CATEGORICAL].groupby(codes, sort=True).nunique().reset_index(drop=True)
    return pd.concat([last, nunique.add_suffix("_nunique")], axis=1)


def main() -> None:
    t0 = time.time()
    st, codes = load()
    boundaries = np.flatnonzero(np.diff(codes)) + 1
    first_idx = np.r_[0, boundaries]
    last_idx = np.r_[boundaries - 1, len(codes) - 1]
    print(f"Loaded {len(st):,} statements for {len(first_idx):,} customers ({time.time() - t0:.0f}s).")

    numeric = [c for c in st.columns if c not in (ID, DATE, *CATEGORICAL)]
    blocks = [pd.DataFrame({ID: st[ID].iloc[last_idx].to_numpy(),
                            "n_statements": np.diff(np.r_[first_idx, len(codes)])})]
    for i in range(0, len(numeric), CHUNK):
        blocks.append(numeric_block(st, numeric[i:i + CHUNK], codes, first_idx, last_idx))
        print(f"  numeric columns {i + 1}-{min(i + CHUNK, len(numeric))} of {len(numeric)} "
              f"({time.time() - t0:.0f}s)")
    blocks.append(categorical_block(st, codes, last_idx))
    agg = pd.concat(blocks, axis=1)

    split = pd.read_parquet(SPLIT_FILE)
    agg = agg.merge(split, on=ID, how="inner", validate="one_to_one")
    assert len(agg) == len(first_idx), "every customer should have a split"

    FEATURES.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(AGG_FILE, index=False)
    n_feat = agg.shape[1] - 3  # minus ID, target, split
    print(f"\n{len(agg):,} customers x {n_feat} features "
          f"({len(numeric)} numeric x 6 + {len(CATEGORICAL)} categorical x 2 + n_statements).")
    print(f"Written to {AGG_FILE} in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
