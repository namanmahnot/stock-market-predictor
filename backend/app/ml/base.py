"""Common interface shared by every forecasting algorithm.

A forecaster is handed the engineered feature frame for every session up to and
including the forecast origin, and returns the cumulative log return it expects
for the next ``horizon`` sessions. The evaluation harness converts those
returns into prices, so all model families are scored on identical quantities.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import joblib
import numpy as np
import pandas as pd


class NotEnoughDataError(ValueError):
    """Raised when a model cannot be fitted on the rows it was given."""


class BaseForecaster(ABC):
    name: ClassVar[str] = "base"
    display_name: ClassVar[str] = "Base"
    family: ClassVar[str] = "baseline"
    description: ClassVar[str] = ""
    #: "high" marks models that are too slow to refit at every walk-forward
    #: origin; the harness fits those once per segment and rolls the origin.
    refit_cost: ClassVar[str] = "low"
    supports_explanations: ClassVar[bool] = False
    #: Minimum fitting rows, on top of the feature warm-up.
    min_train_rows: ClassVar[int] = 60

    def __init__(self, horizon: int = 1, **params: Any) -> None:
        if horizon < 1:
            raise ValueError("horizon must be at least 1 session.")
        self.horizon = int(horizon)
        self.params: dict[str, Any] = dict(params)
        self.fitted_ = False
        self.train_index_: pd.DatetimeIndex | None = None

    # -- lifecycle ---------------------------------------------------------
    @classmethod
    def is_available(cls) -> tuple[bool, str | None]:
        """Whether the optional dependencies of this model are installed."""
        return True, None

    def fit(self, frame: pd.DataFrame) -> "BaseForecaster":
        if len(frame) < self.min_train_rows:
            raise NotEnoughDataError(
                f"{self.display_name} needs at least {self.min_train_rows} usable sessions, "
                f"received {len(frame)}."
            )
        self._fit(frame)
        self.fitted_ = True
        self.train_index_ = frame.index
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Cumulative log returns for h = 1..horizon, from the last row of ``frame``."""
        if not self.fitted_:
            raise RuntimeError(f"{self.display_name} must be fitted before predicting.")
        prediction = np.asarray(self._predict(frame), dtype="float64").reshape(-1)
        if prediction.shape[0] != self.horizon:
            raise RuntimeError(
                f"{self.display_name} returned {prediction.shape[0]} steps, expected {self.horizon}."
            )
        return np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)

    @abstractmethod
    def _fit(self, frame: pd.DataFrame) -> None: ...

    @abstractmethod
    def _predict(self, frame: pd.DataFrame) -> np.ndarray: ...

    # -- metadata ----------------------------------------------------------
    def hyperparameters(self) -> dict[str, Any]:
        return dict(self.params)

    def feature_names(self) -> list[str]:
        return []

    def explain(self, frame: pd.DataFrame) -> dict[str, Any] | None:
        """SHAP-style contributions for the latest prediction, when supported."""
        return None

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "BaseForecaster":
        return joblib.load(Path(path))
