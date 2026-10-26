"""Linear and kernel regression baselines on engineered features."""
from __future__ import annotations

from typing import Any

from app.ml.models.tabular import TabularForecaster


class LinearRegressionForecaster(TabularForecaster):
    """Ridge-regularised linear model on lagged returns and indicators.

    A plain least-squares fit is unstable when indicators are strongly
    correlated, so a small ridge penalty is used; with alpha near zero it is
    ordinary linear regression. Features are standardised on the training rows
    only, never on validation or test rows.
    """

    name = "linear_regression"
    display_name = "Linear Regression"
    family = "statistical"
    description = (
        "Interpretable linear baseline. Uses lagged returns and technical indicators "
        "rather than the calendar date, so it extrapolates a relationship instead of a trend line."
    )
    scale_features = True

    def __init__(self, horizon: int = 1, alpha: float = 1.0, **params: Any) -> None:
        super().__init__(horizon=horizon, alpha=alpha, **params)

    def _build_estimator(self):
        from sklearn.linear_model import Ridge

        return Ridge(alpha=float(self.params.get("alpha", 1.0)), random_state=None)

    def coefficients(self) -> dict[str, list[float]] | None:
        estimator = getattr(self, "estimator_", None)
        if estimator is None:
            return None
        coef = estimator.coef_
        matrix = coef if coef.ndim > 1 else coef.reshape(1, -1)
        return {
            name: [float(matrix[step][i]) for step in range(matrix.shape[0])]
            for i, name in enumerate(self.feature_names())
        }


class SVRForecaster(TabularForecaster):
    """Support vector regression with an RBF kernel (optional comparison)."""

    name = "svr"
    display_name = "Support Vector Regression"
    family = "machine_learning"
    description = (
        "Kernel regression on the same feature set. Included as an optional nonlinear "
        "comparison; it needs standardised inputs and is sensitive to C and epsilon."
    )
    scale_features = True

    def __init__(
        self, horizon: int = 1, C: float = 1.0, epsilon: float = 0.01, gamma: str = "scale", **params: Any
    ) -> None:
        super().__init__(horizon=horizon, C=C, epsilon=epsilon, gamma=gamma, **params)

    def _build_estimator(self):
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.svm import SVR

        base = SVR(
            C=float(self.params.get("C", 1.0)),
            epsilon=float(self.params.get("epsilon", 0.01)),
            gamma=self.params.get("gamma", "scale"),
            kernel="rbf",
        )
        return MultiOutputRegressor(base)
