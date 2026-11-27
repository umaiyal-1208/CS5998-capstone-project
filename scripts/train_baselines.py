"""Train the baseline models on the per-customer summary features.

    python scripts/train_baselines.py

Two models, as set out in the Project Definition:

  Logistic regression  The traditional credit-scoring model and the floor any
                       other model has to clear.
  LightGBM             Gradient-boosted trees, the standard strong model for
                       tabular data. This is the bar the sequence models have
                       to beat.

Each is trained three ways, to compare the ways of handling class imbalance
promised in the definition:

  none          plain training
  class_weight  defaulters weighted up so both classes carry equal total weight
  undersample   non-defaulters randomly dropped from training until the classes
                are equal

Everything is fitted on the training customers. Imputation medians, scaling
and one-hot categories come from training data only, so nothing about the
validation customers reaches the models. LightGBM uses the validation set for
early stopping, which is a model-selection choice and is what that set is for.
The test set is not loaded here at all.

Scores are the competition metric (scripts/metric.py) on the validation
customers, with 95% bootstrap confidence intervals. Validation predictions are
saved to data/preds/ so later models can be compared and blended with them on
exactly the same customers.
"""
from __future__ import annotations

import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from audit_data import CATEGORICAL, GROUPS, ID, TARGET
from build_features import AGG_FILE
from metric import amex_components
from paths import DATA

PREDS = DATA / "preds"
RESULTS = DATA / "results"
SEED = 5998
N_BOOT = 200

CAT_COLS = [f"{c}_last" for c in CATEGORICAL]

LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 100,
    "feature_fraction": 0.3,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "metric": "auc",
    "seed": SEED,
    "verbose": -1,
}


def load() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    # The test set stays untouched: its rows are not even read.
    agg = pd.read_parquet(AGG_FILE, filters=[("split", "!=", "test")])
    features = [c for c in agg.columns if c not in (ID, TARGET, "split")]
    train = agg[agg["split"] == "train"].reset_index(drop=True)
    valid = agg[agg["split"] == "valid"].reset_index(drop=True)
    return train, valid, features


def imbalance(train: pd.DataFrame, how: str) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Return the training rows and sample weights for one imbalance strategy."""
    y = train[TARGET].to_numpy()
    if how == "none":
        return train, None
    if how == "class_weight":
        w = np.where(y == 1, (y == 0).sum() / (y == 1).sum(), 1.0)
        return train, w
    if how == "undersample":
        rng = np.random.default_rng(SEED)
        neg = np.flatnonzero(y == 0)
        keep = np.r_[np.flatnonzero(y == 1), rng.choice(neg, (y == 1).sum(), replace=False)]
        return train.iloc[np.sort(keep)].reset_index(drop=True), None
    raise ValueError(how)


def bootstrap(y: np.ndarray, p: np.ndarray, other: np.ndarray | None = None) -> np.ndarray:
    """Metric on N_BOOT resamples of the validation customers.

    With `other`, returns the paired difference (p minus other) on the same
    resamples, which is the fair way to ask whether one model beats another.
    """
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        s = amex_components(y[i], p[i])["score"]
        if other is not None:
            s -= amex_components(y[i], other[i])["score"]
        out.append(s)
    return np.array(out)


class LogRegPrep:
    """Median imputation, standard scaling and one-hot encoding, all learned
    from the training rows. Written out by hand rather than with sklearn's
    ColumnTransformer so it works in place in float32: the sklearn version
    made several float64 copies of the 321,000 x 1,085 table and ran a 16 GB
    laptop out of memory."""

    def fit(self, rows: pd.DataFrame, numeric: list[str]) -> "LogRegPrep":
        self.numeric = numeric
        x = rows[numeric].to_numpy(np.float32)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            self.median = np.nan_to_num(np.nanmedian(x, axis=0))
            x = np.where(np.isnan(x), self.median, x)
            self.mean = x.mean(axis=0)
            self.std = x.std(axis=0)
        self.std[self.std == 0] = 1.0
        self.onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        self.onehot.fit(rows[CAT_COLS])
        return self

    def transform(self, rows: pd.DataFrame) -> np.ndarray:
        x = rows[self.numeric].to_numpy(np.float32)
        nan = np.isnan(x)
        x[nan] = np.broadcast_to(self.median, x.shape)[nan]
        x -= self.mean
        x /= self.std
        return np.hstack([x, self.onehot.transform(rows[CAT_COLS])])


def fit_logreg(train, valid, features, how):
    rows, w = imbalance(train, how)
    prep = LogRegPrep().fit(rows, [c for c in features if c not in CAT_COLS])
    model = LogisticRegression(max_iter=500, C=0.1)
    model.fit(prep.transform(rows), rows[TARGET].to_numpy(), sample_weight=w)
    return model.predict_proba(prep.transform(valid))[:, 1], None


def shift_categories(df: pd.DataFrame) -> pd.DataFrame:
    """LightGBM treats negative category codes as missing and would merge the
    -1 code into its NaN handling. Shifting every code up by one keeps -1 as a
    category of its own (code 0)."""
    return df.assign(**{c: df[c] + 1 for c in CAT_COLS})


def fit_lightgbm(train, valid, features, how):
    rows, w = imbalance(train, how)
    rows, valid = shift_categories(rows), shift_categories(valid)
    dtrain = lgb.Dataset(rows[features], rows[TARGET], weight=w, categorical_feature=CAT_COLS)
    dvalid = lgb.Dataset(valid[features], valid[TARGET], reference=dtrain)
    model = lgb.train(
        LGB_PARAMS, dtrain, num_boost_round=5000, valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model.predict(valid[features], num_iteration=model.best_iteration), model


def family(col: str) -> str:
    if col == "n_statements":
        return "History length"
    return GROUPS.get(col.split("_")[0], "?")


def importance(model: lgb.Booster) -> None:
    gain = pd.Series(model.feature_importance("gain"), index=model.feature_name())
    share = gain / gain.sum()

    print("\n## LightGBM: which features it relies on\n")
    print("Share of total split gain, best LightGBM model.\n")
    by_family = share.groupby(share.index.map(family)).agg(["sum", "size"])
    by_family = by_family.sort_values("sum", ascending=False).reset_index()
    by_family.columns = ["Family", "Gain share", "Features"]
    by_family["Gain share"] = by_family["Gain share"].map(lambda v: f"{100 * v:.1f}%")
    print(by_family.to_markdown(index=False))

    stat = share.groupby(share.index.map(lambda c: c.rsplit("_", 1)[-1] if c != "n_statements" else "count"))
    by_stat = stat.sum().sort_values(ascending=False).reset_index()
    by_stat.columns = ["Statistic", "Gain share"]
    by_stat["Gain share"] = by_stat["Gain share"].map(lambda v: f"{100 * v:.1f}%")
    print("\n" + by_stat.to_markdown(index=False))

    top = share.sort_values(ascending=False).head(15).reset_index()
    top.columns = ["Feature", "Gain share"]
    top["Gain share"] = top["Gain share"].map(lambda v: f"{100 * v:.1f}%")
    print("\nTop 15 features:\n")
    print(top.to_markdown(index=False))


def main() -> None:
    t0 = time.time()
    train, valid, features = load()
    y = valid[TARGET].to_numpy()
    print(f"Train {len(train):,} customers, validation {len(valid):,}, {len(features)} features.\n")

    PREDS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows, preds, boosters = [], {}, {}
    for name, fit in [("logreg", fit_logreg), ("lightgbm", fit_lightgbm)]:
        for how in ["none", "class_weight", "undersample"]:
            t = time.time()
            p, model = fit(train, valid, features, how)
            key = f"{name}_{how}"
            preds[key], boosters[key] = p, model
            parts = amex_components(y, p)
            lo, hi = np.percentile(bootstrap(y, p), [2.5, 97.5])
            rows.append({
                "Model": name, "Imbalance": how,
                "Score": parts["score"], "95% CI": f"{lo:.4f} - {hi:.4f}",
                "Gini": parts["gini"], "Top 4%": parts["top_four"],
                "Trees": model.best_iteration if model else "-",
                "Minutes": (time.time() - t) / 60,
            })
            pd.DataFrame({ID: valid[ID], TARGET: y, "pred": p}).to_parquet(PREDS / f"{key}_valid.parquet", index=False)
            print(f"  {key:<26} score {parts['score']:.4f}  ({(time.time() - t) / 60:.1f} min)", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "baselines.csv", index=False)
    shown = res.assign(**{c: res[c].map(lambda v: f"{v:.4f}") for c in ["Score", "Gini", "Top 4%"]},
                       Minutes=res["Minutes"].map(lambda v: f"{v:.1f}"))
    print("\n## Validation scores\n")
    print(f"Competition metric on {len(valid):,} validation customers. "
          f"95% CI from {N_BOOT} bootstrap resamples.\n")
    print(shown.to_markdown(index=False))

    best = {m: res[res["Model"] == m].sort_values("Score").iloc[-1] for m in ["logreg", "lightgbm"]}
    k_lgb = f"lightgbm_{best['lightgbm']['Imbalance']}"
    k_lr = f"logreg_{best['logreg']['Imbalance']}"
    diff = bootstrap(y, preds[k_lgb], preds[k_lr])
    lo, hi = np.percentile(diff, [2.5, 97.5])
    print(f"\nBest LightGBM ({k_lgb}) minus best logistic regression ({k_lr}): "
          f"{diff.mean():+.4f}, 95% CI {lo:+.4f} to {hi:+.4f} (paired bootstrap). "
          f"It is above zero in {(diff > 0).mean():.0%} of resamples.")

    importance(boosters[k_lgb])
    boosters[k_lgb].save_model(str(RESULTS / f"{k_lgb}.txt"))
    print(f"\nDone in {(time.time() - t0) / 60:.0f} minutes.")


if __name__ == "__main__":
    main()
