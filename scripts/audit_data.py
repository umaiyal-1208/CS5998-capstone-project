"""Audit the downloaded data: counts, statement histories, labels, dtypes and
missing values.

    python scripts/audit_data.py                     # print the audit
    python scripts/audit_data.py > audit-tables.md   # keep it

The output is markdown tables, so it can go straight into docs/data-audit.md.
One row per feature column is also written to data/audit/columns.csv, so any
number in the audit can be traced back to a column without rerunning.

Missing values need care in this copy of the data. raddar's version stores
columns as integers where possible, and an integer column cannot hold NaN, so
for those columns a missing value may have been encoded as a sentinel such as
-1. The audit therefore reports, for every integer column, how often its minimum
value occurs, so a sentinel shows up as a suspiciously common minimum rather
than being counted as a real value.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from paths import DATA, TRAIN_LABELS, TRAIN_PARQUET

AUDIT = DATA / "audit"

# Figures published for the original competition data, used as the reference
# the working copy is checked against. The "190" often quoted as the feature
# count is the number of columns in train_data.csv, which includes the customer
# ID and the statement date.
PUBLISHED = {"rows": 5_531_451, "customers": 458_913, "columns": 190}

ID, DATE, TARGET = "customer_ID", "S_2", "target"

# The value raddar's integer columns use for a missing entry. sentinels() prints
# the evidence: no integer column holds a NaN, and -1 is the only negative value
# any of them takes.
MISSING_CODE = -1

# Feature columns are named <group>_<number>; the competition documents these groups.
GROUPS = {
    "D": "Delinquency",
    "S": "Spend",
    "P": "Payment",
    "B": "Balance",
    "R": "Risk",
}

# Declared categorical by the competition organisers.
CATEGORICAL = [
    "B_30", "B_38", "D_114", "D_116", "D_117", "D_120",
    "D_126", "D_63", "D_64", "D_66", "D_68",
]


def table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def pct(part: float, whole: float) -> str:
    return f"{100 * part / whole:.1f}%"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in (TRAIN_PARQUET, TRAIN_LABELS):
        if not path.exists():
            sys.exit(f"{path} not found; run scripts/download_data.py first")
    statements = pd.read_parquet(TRAIN_PARQUET)
    labels = pd.read_csv(TRAIN_LABELS)
    return statements, labels


def overview(st: pd.DataFrame, lb: pd.DataFrame, features: list[str]) -> None:
    rows = [
        ("Statement rows", f"{len(st):,}", f"{PUBLISHED['rows']:,}"),
        ("Customers with statements", f"{st[ID].nunique():,}", f"{PUBLISHED['customers']:,}"),
        ("Columns, including ID and date", f"{st.shape[1]}", f"{PUBLISHED['columns']}"),
        ("Feature columns", f"{len(features)}", "-"),
        ("Label rows", f"{len(lb):,}", f"{PUBLISHED['customers']:,}"),
    ]
    print("## Overview\n")
    print(table(pd.DataFrame(rows, columns=["Measure", "This copy", "Published"])))
    mem_gb = st.memory_usage(deep=True).sum() / 1024**3
    print(f"\nIn memory as a pandas DataFrame: {mem_gb:.2f} GB.\n")


def dtypes(st: pd.DataFrame, features: list[str]) -> None:
    counts = st[features].dtypes.astype(str).value_counts().rename_axis("dtype").reset_index(name="columns")
    print("## Column types\n")
    print(table(counts))
    print(f"\nThe date column `{DATE}` is stored as `{st[DATE].dtype}`.\n")


def histories(st: pd.DataFrame, lb: pd.DataFrame) -> None:
    per_customer = st.groupby(ID).size()
    dist = per_customer.value_counts().sort_index()
    defaults = lb.set_index(ID)[TARGET].reindex(per_customer.index)
    rate = defaults.groupby(per_customer).mean()

    out = pd.DataFrame({
        "Statements": dist.index,
        "Customers": [f"{n:,}" for n in dist.values],
        "Share": [pct(n, dist.sum()) for n in dist.values],
        "Default rate": [f"{100 * rate[k]:.1f}%" for k in dist.index],
    })
    print("## Statements per customer\n")
    print(table(out))

    dates = pd.to_datetime(st[DATE])
    print(f"\nStatement dates run from {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}.")
    by_month = dates.dt.to_period("M").value_counts().sort_index()
    print(f"They fall in {len(by_month)} calendar months.\n")
    print(table(pd.DataFrame({"Month": by_month.index.astype(str), "Statements": [f"{n:,}" for n in by_month.values]})))

    dup = st.duplicated([ID, DATE]).sum()
    unsorted = (~st.groupby(ID)[DATE].apply(lambda s: s.is_monotonic_increasing)).sum()
    print(f"\nRows sharing a customer and statement date: {dup:,}.")
    print(f"Customers whose rows are not in date order: {unsorted:,}.\n")


def labels(st: pd.DataFrame, lb: pd.DataFrame) -> None:
    st_ids, lb_ids = set(st[ID].unique()), set(lb[ID])
    rows = [
        ("Unique customers in labels", f"{lb[ID].nunique():,}"),
        ("Duplicate label rows", f"{lb[ID].duplicated().sum():,}"),
        ("Target values", ", ".join(str(v) for v in sorted(lb[TARGET].unique()))),
        ("Default rate", pct(lb[TARGET].sum(), len(lb))),
        ("Customers with statements but no label", f"{len(st_ids - lb_ids):,}"),
        ("Customers with a label but no statements", f"{len(lb_ids - st_ids):,}"),
    ]
    print("## Labels\n")
    print(table(pd.DataFrame(rows, columns=["Measure", "Value"])))
    print()


def columns(st: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """One row per feature column: group, dtype, missing share and integer minimum."""
    n = len(st)
    rows = []
    for col in features:
        s = st[col]
        row = {
            "column": col,
            "group": GROUPS.get(col.split("_")[0], "?"),
            "dtype": str(s.dtype),
            "categorical": col in CATEGORICAL,
            "nan_share": s.isna().mean(),
            "unique": s.nunique(dropna=True),
            "min_value": None,
            "min_share": None,
        }
        if pd.api.types.is_integer_dtype(s):
            lo = s.min()
            row.update(min_value=int(lo), min_share=(s == lo).sum() / n)
        # Integer columns cannot hold NaN, so their missing entries are the -1 code.
        row["missing_share"] = row["min_share"] if row["min_value"] == MISSING_CODE else row["nan_share"]
        rows.append(row)
    return pd.DataFrame(rows)


def groups(cols: pd.DataFrame) -> None:
    g = cols.groupby("group").agg(
        columns=("column", "size"),
        categorical=("categorical", "sum"),
        any_missing=("missing_share", lambda x: (x > 0).sum()),
        median_missing=("missing_share", "median"),
        over_10pct=("missing_share", lambda x: (x > 0.10).sum()),
        over_50pct=("missing_share", lambda x: (x > 0.50).sum()),
    ).reset_index()
    g["median_missing"] = g["median_missing"].map(lambda v: f"{100 * v:.1f}%")
    print("## Feature groups\n")
    print(f"Missing means NaN in a float column or {MISSING_CODE} in an integer column.\n")
    print(table(g.rename(columns={
        "group": "Group", "columns": "Columns", "categorical": "Categorical",
        "any_missing": "With any missing", "median_missing": "Median missing share",
        "over_10pct": ">10% missing", "over_50pct": ">50% missing",
    })))
    m = cols["missing_share"]
    print(f"\nAcross all {len(cols)} features: {(m > 0).sum()} have missing values, "
          f"{(m > 0.10).sum()} are over 10% missing and {(m > 0.50).sum()} are over 50% missing.\n")


def sentinels(cols: pd.DataFrame) -> None:
    ints = cols[cols["min_value"].notna()]
    neg = ints[ints["min_value"] < 0].sort_values("min_share", ascending=False)
    print("## Integer columns and possible missing-value codes\n")
    with_nan = (ints["nan_share"] > 0).sum()
    other = sorted(set(neg["min_value"].astype(int)) - {MISSING_CODE})
    print(f"{len(ints)} feature columns are stored as integers and {with_nan} of them contain a NaN. "
          f"{len(neg)} have a negative minimum, and that minimum is {MISSING_CODE} in every case"
          + (f" except {other}" if other else "") + ".\n")
    if len(neg):
        shown = neg.head(20).assign(
            min_share=lambda d: d["min_share"].map(lambda v: f"{100 * v:.1f}%"),
            min_value=lambda d: d["min_value"].astype(int),
        )[["column", "group", "dtype", "categorical", "min_value", "min_share", "unique"]]
        print(table(shown.rename(columns={
            "column": "Column", "group": "Group", "dtype": "Type", "categorical": "Categorical",
            "min_value": "Minimum", "min_share": "Rows at minimum", "unique": "Distinct values",
        })))
        print()


def most_missing(cols: pd.DataFrame) -> None:
    top = cols.sort_values("missing_share", ascending=False).head(20)
    top = top.assign(missing_share=top["missing_share"].map(lambda v: f"{100 * v:.1f}%"))
    print("## Columns with the most missing values\n")
    print(table(top[["column", "group", "dtype", "missing_share"]].rename(columns={
        "column": "Column", "group": "Group", "dtype": "Type", "missing_share": "Missing share",
    })))
    print()


def categorical(st: pd.DataFrame) -> None:
    rows = []
    for col in CATEGORICAL:
        vc = st[col].value_counts(dropna=False)
        rows.append({
            "Column": col,
            "Type": str(st[col].dtype),
            "Distinct (incl. missing)": len(vc),
            "Values": ", ".join(str(v) for v in sorted(vc.index, key=lambda x: (pd.isna(x), x))[:12]),
        })
    print("## Categorical columns\n")
    print(table(pd.DataFrame(rows)))
    print()


def main() -> None:
    st, lb = load()
    features = [c for c in st.columns if c not in (ID, DATE)]

    overview(st, lb, features)
    dtypes(st, features)
    labels(st, lb)
    histories(st, lb)

    cols = columns(st, features)
    groups(cols)
    sentinels(cols)
    most_missing(cols)
    categorical(st)

    AUDIT.mkdir(parents=True, exist_ok=True)
    cols.to_csv(AUDIT / "columns.csv", index=False)
    print(f"Per-column details written to {AUDIT / 'columns.csv'}.")


if __name__ == "__main__":
    main()
