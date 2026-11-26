"""Tests for scripts/metric.py.

    python -m pytest tests -q
"""
import numpy as np
import pytest

from metric import amex_components, amex_metric


def reference_metric(y_true, y_pred):
    """The reference implementation the competition hosts published, kept as
    close to the original as possible so ours can be checked against it."""
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:, 0] == 0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:, 0]) / np.sum(labels[:, 0])

    gini = [0, 0]
    for i in [1, 0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:, 0] == 0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] * weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1] / gini[0] + top_four)


def test_hand_worked_example():
    """Four customers, ranked: default, non-default, default, non-default.

    Weights are 1, 20, 1, 20, total 42. The top 4% is int(0.04 * 42) = 1 unit
    of weight, which holds only the first customer: 1 of the 2 defaults, so
    D = 0.5.

    Model ranking: cumulative weight share 1/42, 21/42, 22/42, 1 and share of
    defaults found 0.5, 0.5, 1, 1. Sum of (found - weight share) * weight:
        (0.5 - 1/42) * 1 + 0 * 20 + (1 - 22/42) * 1 + 0 * 20 = 40/42
    Perfect ranking (both defaults first): weight shares 1/42, 2/42, 22/42, 1,
    found 0.5, 1, 1, 1:
        (0.5 - 1/42) + (1 - 2/42) + (1 - 22/42) * 20 + 0 = 460/42
    So G = 40/460 = 2/23 and M = 0.5 * (2/23 + 0.5).
    """
    parts = amex_components([1, 0, 1, 0], [0.9, 0.8, 0.3, 0.1])
    assert parts["top_four"] == pytest.approx(0.5)
    assert parts["gini"] == pytest.approx(2 / 23)
    assert parts["score"] == pytest.approx(0.5 * (2 / 23 + 0.5))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference(seed):
    rng = np.random.default_rng(seed)
    y = (rng.random(20_000) < 0.26).astype(int)
    p = y * 0.3 + rng.random(len(y))  # informative but imperfect
    assert amex_metric(y, p) == pytest.approx(reference_metric(y, p), rel=1e-9)


def test_perfect_ranking_has_gini_one():
    rng = np.random.default_rng(3)
    y = (rng.random(5_000) < 0.26).astype(int)
    assert amex_components(y, y + rng.random(len(y)) * 0.5)["gini"] == pytest.approx(1.0)


def test_random_ranking_scores_near_zero_gini():
    rng = np.random.default_rng(4)
    y = (rng.random(200_000) < 0.26).astype(int)
    assert abs(amex_components(y, rng.random(len(y)))["gini"]) < 0.02


def test_only_the_ranking_matters():
    rng = np.random.default_rng(5)
    y = (rng.random(5_000) < 0.26).astype(int)
    p = y * 0.3 + rng.random(len(y))
    assert amex_metric(y, p) == pytest.approx(amex_metric(y, np.exp(5 * p)))


def test_rejects_single_class():
    with pytest.raises(ValueError):
        amex_metric([0, 0, 0], [0.1, 0.2, 0.3])
