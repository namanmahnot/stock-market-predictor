"""XGBoost gradient boosting on technical features (SRS 6.1)."""
from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.ml.models.tabular import TabularForecaster


class XGBoostForecaster(TabularForecaster):
    """Gradient-boosted trees, one booster per horizon step.

    A separate booster per step keeps every forecast a direct prediction and
    lets SHAP explain the next-session model on its own.
    """

    name = "xgboost"
    display_name = "XGBoost"
    family = "machine_learning"
    description = (
        "Gradient boosting over the same engineered features as the Random Forest. "
        "Shallow trees and a low learning rate are used because daily return data is "
        "noisy and easy to overfit."
    )
    supports_explanations = True

    def __init__(
        self,
        horizon: int = 1,
        n_estimators: int = 300,
        max_depth: int = 3,
        learning_rate: float = 0.03,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        **params: Any,
    ) -> None:
        super().__init__(
            horizon=horizon,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            **params,
        )

    @classmethod
    def is_available(cls) -> tuple[bool, str | None]:
        try:
            import xgboost  # noqa: F401
        except ImportError:
            return False, "xgboost is not installed (pip install xgboost)."
        return True, None

    def _build_estimator(self):
        from sklearn.multioutput import MultiOutputRegressor
        from xgboost import XGBRegressor

        base = XGBRegressor(
            n_estimators=int(self.params.get("n_estimators", 300)),
            max_depth=int(self.params.get("max_depth", 3)),
            learning_rate=float(self.params.get("learning_rate", 0.03)),
            subsample=float(self.params.get("subsample", 0.8)),
            colsample_bytree=float(self.params.get("colsample_bytree", 0.8)),
            reg_lambda=float(self.params.get("reg_lambda", 1.0)),
            objective="reg:squarederror",
            random_state=settings.random_seed,
            n_jobs=2,
            verbosity=0,
        )
        if self.horizon == 1:
            return base
        return MultiOutputRegressor(base)

    def _explain_estimator(self):
        estimator = getattr(self, "estimator_", None)
        inner = getattr(estimator, "estimators_", None)
        # MultiOutputRegressor keeps one booster per horizon step; step 1 first.
        return inner[0] if inner else estimator

    def feature_importances(self) -> dict[str, float] | None:
        estimator = self._explain_estimator()
        if estimator is None or not hasattr(estimator, "feature_importances_"):
            return None
        return {
            name: float(value)
            for name, value in zip(self.feature_names(), estimator.feature_importances_)
        }
