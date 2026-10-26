"""Random Forest regressor on technical features (SRS 6.1)."""
from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.ml.models.tabular import TabularForecaster


class RandomForestForecaster(TabularForecaster):
    """Bagged decision trees, fitted natively on all horizon steps at once."""

    name = "random_forest"
    display_name = "Random Forest"
    family = "machine_learning"
    description = (
        "Ensemble of decision trees over lagged returns, moving-average ratios, RSI, "
        "MACD, volatility and volume features. Captures nonlinear interactions and "
        "exposes feature importance."
    )
    supports_explanations = True

    def __init__(
        self,
        horizon: int = 1,
        n_estimators: int = 300,
        max_depth: int | None = 8,
        min_samples_leaf: int = 5,
        max_features: str | float = "sqrt",
        **params: Any,
    ) -> None:
        super().__init__(
            horizon=horizon,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            **params,
        )

    def _build_estimator(self):
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=int(self.params.get("n_estimators", 300)),
            max_depth=self.params.get("max_depth", 8),
            min_samples_leaf=int(self.params.get("min_samples_leaf", 5)),
            max_features=self.params.get("max_features", "sqrt"),
            random_state=settings.random_seed,
            n_jobs=-1,
        )

    def feature_importances(self) -> dict[str, float] | None:
        estimator = getattr(self, "estimator_", None)
        if estimator is None or not hasattr(estimator, "feature_importances_"):
            return None
        return {
            name: float(value)
            for name, value in zip(self.feature_names(), estimator.feature_importances_)
        }
