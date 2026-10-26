"""Forecast error metrics (SRS section 9).

All metrics are computed on price levels so that different model families stay
comparable, and every metric is reported next to the naive baseline it has to
beat before it can be called useful.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

EPSILON = 1e-12


@dataclass
class MetricSet:
    n_observations: int
    mae: float | None
    rmse: float | None
    mape: float | None
    directional_accuracy: float | None
    r2: float | None

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    diff = np.asarray(actual) - np.asarray(predicted)
    return float(np.sqrt(np.mean(diff**2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    """Mean absolute percentage error, skipping actuals at or near zero.

    MAPE is unstable for small denominators, so it is reported alongside MAE
    and RMSE rather than on its own.
    """
    actual = np.asarray(actual, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")
    usable = np.abs(actual) > EPSILON
    if not usable.any():
        return None
    return float(np.mean(np.abs((actual[usable] - predicted[usable]) / actual[usable])) * 100.0)


def directional_accuracy(
    actual: np.ndarray, predicted: np.ndarray, reference: np.ndarray
) -> float | None:
    """Share of forecasts whose predicted direction matched the realised one.

    ``reference`` is the last observed price at each forecast origin, so the
    direction is measured against what was actually known at prediction time.

    Pairs where either side is exactly flat are excluded: a session that closed
    unchanged has no direction to predict, and a forecast of no change makes no
    directional claim. The naive baseline is flat by construction, so its
    directional accuracy is reported as undefined rather than as 0%.
    """
    actual = np.asarray(actual, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")
    reference = np.asarray(reference, dtype="float64")
    actual_move = np.sign(actual - reference)
    predicted_move = np.sign(predicted - reference)
    usable = (actual_move != 0) & (predicted_move != 0)
    if not usable.any():
        return None
    return float(np.mean(actual_move[usable] == predicted_move[usable]))


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    """R^2 against the mean of the evaluation window; may be negative."""
    actual = np.asarray(actual, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")
    if actual.size < 2:
        return None
    total = float(np.sum((actual - actual.mean()) ** 2))
    if total <= EPSILON:
        return None
    residual = float(np.sum((actual - predicted) ** 2))
    return float(1.0 - residual / total)


def evaluate(
    actual: np.ndarray, predicted: np.ndarray, reference: np.ndarray | None = None
) -> MetricSet:
    actual = np.asarray(actual, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")
    if actual.size == 0:
        return MetricSet(0, None, None, None, None, None)
    return MetricSet(
        n_observations=int(actual.size),
        mae=mae(actual, predicted),
        rmse=rmse(actual, predicted),
        mape=mape(actual, predicted),
        directional_accuracy=(
            directional_accuracy(actual, predicted, reference) if reference is not None else None
        ),
        r2=r2_score(actual, predicted),
    )


def skill_vs_naive(model_mae: float | None, naive_mae: float | None) -> float | None:
    """Ratio of model MAE to naive MAE: below 1.0 means the model adds value."""
    if model_mae is None or naive_mae is None or naive_mae <= EPSILON:
        return None
    return float(model_mae / naive_mae)
