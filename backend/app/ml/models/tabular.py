"""Shared plumbing for models that learn from engineered tabular features."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.ml.base import BaseForecaster, NotEnoughDataError
from app.ml.features import FEATURE_COLUMNS, build_supervised


class TabularForecaster(BaseForecaster):
    """Fits one multi-output regressor on lagged returns and indicators.

    The estimator predicts the cumulative log return for every horizon step at
    once, which keeps a 30-session forecast a single direct prediction instead
    of a chain of recursive guesses that compounds its own error.
    """

    scale_features: bool = False

    def _build_estimator(self):  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def _fit(self, frame: pd.DataFrame) -> None:
        features, targets = build_supervised(frame, self.horizon, drop_incomplete=True)
        if len(features) < self.min_train_rows:
            raise NotEnoughDataError(
                f"{self.display_name} needs {self.min_train_rows} rows with a complete "
                f"{self.horizon}-session future, only {len(features)} are available."
            )
        self.feature_names_ = list(features.columns)
        x = features.to_numpy(dtype="float64")
        y = targets.to_numpy(dtype="float64")
        if y.shape[1] == 1:
            # A single-step horizon is an ordinary single-output regression.
            y = y.ravel()

        if self.scale_features:
            self.x_mean_ = x.mean(axis=0)
            self.x_std_ = np.where(x.std(axis=0) < 1e-12, 1.0, x.std(axis=0))
            x = (x - self.x_mean_) / self.x_std_

        self.estimator_ = self._build_estimator()
        self.estimator_.fit(x, y)

    def _transform(self, frame: pd.DataFrame) -> np.ndarray:
        # Always the columns this model was fitted on, in that order.
        row = frame[self.feature_names_].iloc[[-1]].to_numpy(dtype="float64")
        if self.scale_features:
            row = (row - self.x_mean_) / self.x_std_
        return row

    def _predict(self, frame: pd.DataFrame) -> np.ndarray:
        prediction = np.asarray(self.estimator_.predict(self._transform(frame)), dtype="float64")
        return prediction.reshape(-1)[: self.horizon]

    def feature_names(self) -> list[str]:
        return list(getattr(self, "feature_names_", FEATURE_COLUMNS))

    def _explain_estimator(self):
        """The estimator SHAP should inspect; subclasses narrow this to h = 1."""
        return getattr(self, "estimator_", None)

    def explain(self, frame: pd.DataFrame) -> dict[str, Any] | None:
        """Feature contributions to the next-session prediction via SHAP.

        These are contributions of the model, not evidence that a feature moves
        the market (SRS 10.3).
        """
        if not self.supports_explanations or not self.fitted_:
            return None
        try:
            import shap
        except ImportError:
            return None

        estimator = self._explain_estimator()
        row = self._transform(frame)
        try:
            explainer = shap.TreeExplainer(estimator)
            raw = np.asarray(explainer.shap_values(row), dtype="float64")
            # Multi-output estimators return one column of values per horizon
            # step; the explanation describes the next session only.
            values = raw[0, :, 0] if raw.ndim == 3 else raw.reshape(1, -1)[0]
            base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        except Exception:
            return None

        names = self.feature_names()
        contributions = [
            {"feature": name, "value": float(row[0][i]), "contribution": float(values[i])}
            for i, name in enumerate(names[: values.size])
        ]
        contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        return {
            "method": "shap.TreeExplainer",
            "target": "cumulative log return, next session",
            "base_value": base_value,
            "prediction": base_value + float(values.sum()),
            "contributions": contributions,
            "note": (
                "Contributions explain this model's output on these features. "
                "They do not establish a causal effect on the market."
            ),
        }
