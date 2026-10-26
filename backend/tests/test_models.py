"""Every registered algorithm must fit, predict and stay reproducible."""
from __future__ import annotations

import numpy as np
import pytest

from app.ml.base import NotEnoughDataError
from app.ml.registry import (
    DEFAULT_MODEL_SET,
    MODEL_REGISTRY,
    UnknownModelError,
    create_model,
    describe_models,
)

# The recurrent models are the slow ones; they are covered separately.
FAST_MODELS = ["naive", "linear_regression", "arima", "random_forest", "xgboost", "svr"]


def test_every_default_model_is_registered():
    for name in DEFAULT_MODEL_SET:
        assert name in MODEL_REGISTRY


def test_catalogue_reports_availability():
    catalogue = {entry["name"]: entry for entry in describe_models()}
    assert set(catalogue) == set(MODEL_REGISTRY)
    for entry in catalogue.values():
        assert entry["available"], entry["unavailable_reason"]


def test_unknown_model_is_rejected():
    with pytest.raises(UnknownModelError):
        create_model("pickle_rick")


@pytest.mark.parametrize("name", FAST_MODELS)
def test_model_fits_and_predicts_the_requested_horizon(name, feature_frame):
    horizon = 3
    model = create_model(name, horizon=horizon)
    model.fit(feature_frame.iloc[:500])
    prediction = model.predict(feature_frame.iloc[:520])
    assert prediction.shape == (horizon,)
    assert np.isfinite(prediction).all()
    # Predictions are cumulative log returns; a few percent per session is the
    # plausible range for daily equity data.
    assert np.abs(prediction).max() < 1.0


@pytest.mark.parametrize("name", ["lstm", "gru"])
def test_recurrent_models_fit_and_predict(name, feature_frame):
    model = create_model(name, horizon=2, epochs=6, hidden_size=8, patience=3)
    model.fit(feature_frame.iloc[:400])
    prediction = model.predict(feature_frame.iloc[:420])
    assert prediction.shape == (2,)
    assert np.isfinite(prediction).all()


def test_naive_predicts_no_change(feature_frame):
    model = create_model("naive", horizon=4).fit(feature_frame)
    assert np.allclose(model.predict(feature_frame), 0.0)


def test_predicting_before_fitting_is_an_error(feature_frame):
    model = create_model("random_forest", horizon=1)
    with pytest.raises(RuntimeError):
        model.predict(feature_frame)


def test_fitting_on_too_few_rows_is_reported(feature_frame):
    model = create_model("random_forest", horizon=1)
    with pytest.raises(NotEnoughDataError):
        model.fit(feature_frame.iloc[:20])


@pytest.mark.parametrize("name", ["random_forest", "xgboost", "linear_regression"])
def test_models_are_reproducible_with_the_same_seed(name, feature_frame):
    """NFR-08: an identical configuration on identical rows gives identical output."""
    window = feature_frame.iloc[:450]
    first = create_model(name, horizon=2).fit(window).predict(window)
    second = create_model(name, horizon=2).fit(window).predict(window)
    np.testing.assert_allclose(first, second, rtol=1e-10, atol=1e-12)


def test_tree_models_expose_shap_contributions(feature_frame):
    model = create_model("xgboost", horizon=1, n_estimators=40).fit(feature_frame.iloc[:400])
    explanation = model.explain(feature_frame.iloc[:400])
    assert explanation is not None
    assert explanation["contributions"]
    top = explanation["contributions"][0]
    assert {"feature", "value", "contribution"} <= set(top)
    # Contributions are ordered by absolute size.
    sizes = [abs(item["contribution"]) for item in explanation["contributions"]]
    assert sizes == sorted(sizes, reverse=True)


def test_arima_selects_an_order_by_aic(feature_frame):
    model = create_model("arima", horizon=1).fit(feature_frame.iloc[:400])
    hyperparameters = model.hyperparameters()
    assert hyperparameters["order"] is not None
    assert "aic" in hyperparameters


def test_models_can_be_saved_and_reloaded(feature_frame, tmp_path):
    from app.ml.base import BaseForecaster

    window = feature_frame.iloc[:400]
    model = create_model("random_forest", horizon=2, n_estimators=30).fit(window)
    path = model.save(tmp_path / "rf.joblib")
    reloaded = BaseForecaster.load(path)
    np.testing.assert_allclose(reloaded.predict(window), model.predict(window))


def test_feature_names_are_recorded_for_tabular_models(feature_frame):
    model = create_model("random_forest", horizon=1, n_estimators=30).fit(feature_frame.iloc[:400])
    names = model.feature_names()
    assert len(names) == 24
    assert "rsi_14" in names
