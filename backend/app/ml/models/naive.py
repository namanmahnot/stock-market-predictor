"""Naive last-value forecast: the baseline every other model must beat."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.base import BaseForecaster


class NaiveForecaster(BaseForecaster):
    """Random-walk forecast: tomorrow's price equals the last observed close.

    Expressed in the shared target space this is a cumulative log return of
    zero at every horizon step. It needs no training, and for daily equity
    prices it is a genuinely strong reference point.
    """

    name = "naive"
    display_name = "Naive (last value)"
    family = "baseline"
    description = (
        "Carries the most recent adjusted close forward unchanged. Any model that "
        "cannot beat this on unseen data adds no predictive value."
    )
    min_train_rows = 5

    def _fit(self, frame: pd.DataFrame) -> None:
        return None

    def _predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.zeros(self.horizon, dtype="float64")
