"""Split customers into train, validation and test sets: 70/15/15, stratified.

    python scripts/make_split.py                # write the split and print a summary
    python scripts/make_split.py > split.md     # keep the summary

The split is by customer, never by statement row. A customer's 13 monthly
statements are strongly correlated, so if some of them landed in training and
others in test, the test score would measure memory of that customer rather than
prediction for a new one.

Stratifying on the target keeps the default rate the same in all three sets.
Within each class the customers are shuffled with a fixed seed, so the split is
identical on every run and every machine. It is written to
data/splits/split.parquet (customer_ID, target, split). That file is not
committed, because customer IDs are part of the competition data, which may not
be redistributed. Instead the summary prints a SHA-256 fingerprint of each set's
sorted IDs. Anyone who reruns this script can confirm they got the same split.

The test set is not to be touched until the final evaluation. Model selection
and tuning use the validation set only.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from paths import DATA, TRAIN_LABELS, TRAIN_PARQUET

SPLITS = DATA / "splits"
SPLIT_FILE = SPLITS / "split.parquet"

SEED = 5998
SHARES = {"train": 0.70, "valid": 0.15, "test": 0.15}
ID, TARGET = "customer_ID", "target"


def assign(labels: pd.DataFrame) -> pd.Series:
    """Give each customer a split name, stratified on the target."""
    rng = np.random.default_rng(SEED)
    split = pd.Series("", index=labels.index, dtype=object)
    for _, group in labels.groupby(TARGET):
        idx = rng.permutation(group.index.to_numpy())
        n_train = round(len(idx) * SHARES["train"])
        n_valid = round(len(idx) * SHARES["valid"])
        split[idx[:n_train]] = "train"
        split[idx[n_train:n_train + n_valid]] = "valid"
        split[idx[n_train + n_valid:]] = "test"
    return split


def fingerprint(ids: pd.Series) -> str:
    joined = "\n".join(sorted(ids)).encode()
    return hashlib.sha256(joined).hexdigest()[:16]


def summary(df: pd.DataFrame) -> None:
    order = list(SHARES)
    rows = []
    for name in order:
        part = df[df["split"] == name]
        rows.append({
            "Split": name,
            "Customers": f"{len(part):,}",
            "Share": f"{100 * len(part) / len(df):.1f}%",
            "Defaults": f"{part[TARGET].sum():,}",
            "Default rate": f"{100 * part[TARGET].mean():.2f}%",
            "Statements": f"{part['statements'].sum():,}",
            "Full 13-month histories": f"{100 * (part['statements'] == 13).mean():.1f}%",
            "Fingerprint": fingerprint(part[ID]),
        })
    print("## Customer split\n")
    print(f"Seed {SEED}. Shares {', '.join(f'{k} {v:.0%}' for k, v in SHARES.items())}, "
          f"stratified on `{TARGET}`.\n")
    print(pd.DataFrame(rows).to_markdown(index=False))

    # History length is a strong signal (see docs/data-audit.md), so check that
    # stratifying on the target alone has also kept it balanced.
    dist = pd.crosstab(df["statements"], df["split"], normalize="columns")[order]
    dist = dist.map(lambda v: f"{100 * v:.1f}%").reset_index()
    print("\n## Statements per customer, by split\n")
    print(dist.rename(columns={"statements": "Statements"}).to_markdown(index=False))

    sets = {name: set(df.loc[df["split"] == name, ID]) for name in order}
    overlap = sum(len(sets[a] & sets[b]) for i, a in enumerate(order) for b in order[i + 1:])
    print(f"\nCustomers appearing in more than one split: {overlap}.")
    print(f"Customers without a split: {(df['split'] == '').sum()}.\n")


def main() -> None:
    labels = pd.read_csv(TRAIN_LABELS)
    counts = pd.read_parquet(TRAIN_PARQUET, columns=[ID])[ID].value_counts()
    labels["statements"] = labels[ID].map(counts).astype(int)
    labels["split"] = assign(labels)

    summary(labels)

    SPLITS.mkdir(parents=True, exist_ok=True)
    labels[[ID, TARGET, "split"]].to_parquet(SPLIT_FILE, index=False)
    print(f"Split written to {SPLIT_FILE}.")


if __name__ == "__main__":
    main()
