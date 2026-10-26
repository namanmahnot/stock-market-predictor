"""Metric and split arithmetic, verified against hand calculations."""
from __future__ import annotations

import numpy as np
import pytest

from app.ml.metrics import (
    directional_accuracy,
    evaluate,
    mae,
    mape,
    r2_score,
    rmse,
    skill_vs_naive,
)
from app.ml.splits import chronological_split, walk_forward_origins


def test_mae_and_rmse_against_known_values():
    actual = np.array([10.0, 12.0, 14.0])
    predicted = np.array([11.0, 10.0, 14.0])
    assert mae(actual, predicted) == pytest.approx(1.0)
    assert rmse(actual, predicted) == pytest.approx(np.sqrt(5.0 / 3.0))


def test_mape_is_a_percentage():
    actual = np.array([100.0, 200.0])
    predicted = np.array([110.0, 180.0])
    assert mape(actual, predicted) == pytest.approx(10.0)


def test_mape_ignores_zero_actuals():
    assert mape(np.array([0.0, 50.0]), np.array([1.0, 55.0])) == pytest.approx(10.0)
    assert mape(np.array([0.0]), np.array([1.0])) is None


def test_directional_accuracy_counts_matching_moves():
    reference = np.array([100.0, 100.0, 100.0, 100.0])
    actual = np.array([101.0, 99.0, 102.0, 98.0])
    predicted = np.array([103.0, 97.0, 99.0, 99.0])  # right, right, wrong, right
    assert directional_accuracy(actual, predicted, reference) == pytest.approx(0.75)


def test_directional_accuracy_is_undefined_for_a_flat_forecast():
    reference = np.array([100.0, 100.0])
    actual = np.array([101.0, 99.0])
    predicted = reference.copy()
    assert directional_accuracy(actual, predicted, reference) is None


def test_r2_is_one_for_a_perfect_fit_and_negative_for_a_bad_one():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_score(actual, actual) == pytest.approx(1.0)
    assert r2_score(actual, np.array([10.0, 10.0, 10.0, 10.0])) < 0


def test_evaluate_bundles_every_metric():
    reference = np.array([100.0, 100.0])
    result = evaluate(np.array([101.0, 102.0]), np.array([100.5, 103.0]), reference)
    assert result.n_observations == 2
    assert result.mae == pytest.approx(0.75)
    assert result.directional_accuracy == pytest.approx(1.0)


def test_evaluate_handles_an_empty_window():
    result = evaluate(np.array([]), np.array([]))
    assert result.n_observations == 0 and result.mae is None


def test_skill_vs_naive_is_a_ratio():
    assert skill_vs_naive(2.0, 4.0) == pytest.approx(0.5)
    assert skill_vs_naive(2.0, 0.0) is None
    assert skill_vs_naive(None, 4.0) is None


def test_chronological_split_is_ordered_and_disjoint():
    split = chronological_split(1000, 0.7, 0.15)
    assert split.describe() == {"n_rows": 1000, "n_train": 700, "n_validation": 150, "n_test": 150}
    assert split.train[-1] < split.validation[0] < split.validation[-1] < split.test[0]
    assert set(split.train).isdisjoint(split.validation)
    assert set(split.validation).isdisjoint(split.test)
    assert len(split.train) + len(split.validation) + len(split.test) == 1000


def test_chronological_split_always_leaves_rows_for_every_segment():
    split = chronological_split(12, 0.9, 0.05)
    assert len(split.train) >= 1 and len(split.validation) >= 1 and len(split.test) >= 1


def test_chronological_split_rejects_impossible_requests():
    with pytest.raises(ValueError):
        chronological_split(5)
    with pytest.raises(ValueError):
        chronological_split(100, 0.9, 0.2)


def test_walk_forward_origins_thins_and_keeps_the_most_recent():
    origins = walk_forward_origins(range(0, 100), step=5)
    assert origins[:3] == [0, 5, 10]
    capped = walk_forward_origins(range(0, 100), step=1, max_origins=10)
    assert capped == list(range(90, 100))


def test_walk_forward_origins_rejects_a_zero_step():
    with pytest.raises(ValueError):
        walk_forward_origins(range(0, 10), step=0)
