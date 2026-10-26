"""Whitelist of forecasting algorithms exposed by the API.

Models are always resolved through this registry, so a request can never load
an arbitrary class or artifact path (SRS section 16, item 8).
"""
from __future__ import annotations

from typing import Any

from app.ml.base import BaseForecaster
from app.ml.models.arima import ArimaForecaster
from app.ml.models.linear import LinearRegressionForecaster, SVRForecaster
from app.ml.models.naive import NaiveForecaster
from app.ml.models.random_forest import RandomForestForecaster
from app.ml.models.recurrent import GRUForecaster, LSTMForecaster
from app.ml.models.xgboost_model import XGBoostForecaster

MODEL_REGISTRY: dict[str, type[BaseForecaster]] = {
    NaiveForecaster.name: NaiveForecaster,
    LinearRegressionForecaster.name: LinearRegressionForecaster,
    ArimaForecaster.name: ArimaForecaster,
    RandomForestForecaster.name: RandomForestForecaster,
    XGBoostForecaster.name: XGBoostForecaster,
    LSTMForecaster.name: LSTMForecaster,
    GRUForecaster.name: GRUForecaster,
    SVRForecaster.name: SVRForecaster,
}

#: The initial model set recommended in SRS section 6.2.
DEFAULT_MODEL_SET: list[str] = [
    "naive",
    "linear_regression",
    "arima",
    "random_forest",
    "xgboost",
    "lstm",
]

BASELINE_MODEL = "naive"


class UnknownModelError(ValueError):
    pass


def get_model_class(name: str) -> type[BaseForecaster]:
    key = (name or "").strip().lower()
    if key not in MODEL_REGISTRY:
        raise UnknownModelError(
            f"Unknown model '{name}'. Supported models: {', '.join(sorted(MODEL_REGISTRY))}."
        )
    return MODEL_REGISTRY[key]


def create_model(name: str, horizon: int = 1, **params: Any) -> BaseForecaster:
    model_class = get_model_class(name)
    available, reason = model_class.is_available()
    if not available:
        raise UnknownModelError(f"Model '{name}' is unavailable: {reason}")
    return model_class(horizon=horizon, **params)


def describe_models() -> list[dict[str, Any]]:
    """Catalogue for the frontend model selector."""
    catalogue = []
    for name, model_class in MODEL_REGISTRY.items():
        available, reason = model_class.is_available()
        catalogue.append(
            {
                "name": name,
                "display_name": model_class.display_name,
                "family": model_class.family,
                "description": model_class.description,
                "available": available,
                "unavailable_reason": reason,
                "in_default_set": name in DEFAULT_MODEL_SET,
                "supports_explanations": model_class.supports_explanations,
                "refit_cost": model_class.refit_cost,
                "min_train_rows": model_class.min_train_rows,
            }
        )
    catalogue.sort(key=lambda item: (not item["in_default_set"], item["name"]))
    return catalogue
