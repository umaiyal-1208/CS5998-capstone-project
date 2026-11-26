"""The American Express competition metric.

    M = 0.5 * (G + D)

G is the normalised Gini coefficient and D is the share of all defaults that
land in the 4% of customers the model ranks as riskiest. Both are computed with
every non-default weighted 20 times. The organisers subsampled non-defaults to
5% when building the data, so the weight of 20 restores the real class balance;
without it, scores here would not be comparable with published results.

Only the ranking of the predictions matters, not their values, so the metric
says nothing about calibration. Calibration is checked separately.

tests/test_metric.py checks this against a small example worked out by hand and
against the reference code the competition hosts published.
"""
from __future__ import annotations

import numpy as np

NEGATIVE_WEIGHT = 20.0
TOP_SHARE = 0.04


def _weights(y: np.ndarray) -> np.ndarray:
    return np.where(y == 0, NEGATIVE_WEIGHT, 1.0)


def _gini(y: np.ndarray, order: np.ndarray) -> float:
    """Weighted Gini of the labels when customers are taken in `order`."""
    y = y[order]
    w = _weights(y)
    random = np.cumsum(w) / w.sum()          # share of weighted customers seen so far
    lorentz = np.cumsum(y * w) / np.sum(y * w)  # share of defaults found so far
    return float(np.sum((lorentz - random) * w))


def _top_four(y: np.ndarray, order: np.ndarray) -> float:
    """Share of all defaults found in the riskiest 4% (by weight) of customers."""
    y = y[order]
    w = _weights(y)
    cutoff = np.cumsum(w) <= int(TOP_SHARE * w.sum())
    return float(y[cutoff].sum() / y.sum())


def amex_components(y_true, y_pred) -> dict[str, float]:
    """Return the metric and its two parts: gini, top_four and score."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-D arrays of the same length")
    if not np.isin(y, (0, 1)).all() or y.sum() in (0, len(y)):
        raise ValueError("y_true must be 0/1 and contain both classes")

    by_pred = np.argsort(p, kind="stable")[::-1]  # riskiest first
    by_true = np.argsort(y, kind="stable")[::-1]  # a perfect ranking
    gini = _gini(y, by_pred) / _gini(y, by_true)
    top_four = _top_four(y, by_pred)
    return {"gini": gini, "top_four": top_four, "score": 0.5 * (gini + top_four)}


def amex_metric(y_true, y_pred) -> float:
    return amex_components(y_true, y_pred)["score"]
