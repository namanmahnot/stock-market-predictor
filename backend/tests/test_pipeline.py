"""Pipeline behaviour: chronology, leakage, selection rules and forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml import pipeline
from app.ml.base import BaseForecaster
from app.ml.pipeline import (
    InsufficientHistoryError,
    ModelReport,
    leaderboard_rows,
    prepare_dataset,
    run_experiment,
    select_best_model,
    walk_forward_evaluate,
)
from app.ml.registry import MODEL_REGISTRY
from app.ml.splits import chronological_split


@pytest.fixture(scope="module")
def experiment(request):
    """One experiment reused by several assertions (training is not cheap)."""
    from datetime import datetime, timezone

    from app.data.providers import PriceHistory
    from tests.conftest import make_price_frame

    frame = make_price_frame()
    history = PriceHistory(
        ticker="TEST",
        frame=frame,
        source="csv",
        retrieved_at=datetime.now(timezone.utc),
        name="Synthetic Test Series",
        currency="USD",
    )
    return run_experiment(history, horizon=3, model_names=["naive", "linear_regression"])


def test_prepare_dataset_rejects_too_little_history():
    from datetime import datetime, timezone

    from app.data.providers import PriceHistory
    from tests.conftest import make_price_frame

    short = PriceHistory("TEST", make_price_frame(120), "csv", datetime.now(timezone.utc))
    with pytest.raises(InsufficientHistoryError):
        prepare_dataset(short)


def test_experiment_reports_every_requested_model(experiment):
    names = {report.name for report in experiment.reports}
    assert names == {"naive", "linear_regression"}
    assert all(report.ok for report in experiment.reports)


def test_experiment_splits_are_chronological_and_contiguous(experiment):
    dates = experiment.split_dates
    assert dates["train_start"] < dates["train_end"] < dates["validation_start"]
    assert dates["validation_start"] < dates["validation_end"] < dates["test_start"]
    assert dates["test_start"] < dates["test_end"]
    described = experiment.split.describe()
    assert described["n_train"] + described["n_validation"] + described["n_test"] == described["n_rows"]
    assert described["n_rows"] == experiment.feature_rows


def test_forecast_has_one_step_per_horizon_session(experiment):
    report = experiment.report_for("linear_regression")
    assert len(report.forecast) == 3
    assert [step.step for step in report.forecast] == [1, 2, 3]
    dates = [step.target_date for step in report.forecast]
    assert dates == sorted(dates)
    assert all(date > experiment.reference_date for date in dates)
    assert all(step.predicted_price > 0 for step in report.forecast)


def test_forecast_intervals_bracket_the_prediction(experiment):
    for report in experiment.reports:
        for step in report.forecast:
            if step.lower_bound is None:
                continue
            assert step.lower_bound < step.predicted_price < step.upper_bound


def test_naive_forecast_repeats_the_last_close(experiment):
    report = experiment.report_for("naive")
    for step in report.forecast:
        assert step.predicted_price == pytest.approx(experiment.reference_price)
        assert step.change == pytest.approx(0.0)


def test_both_segments_are_evaluated_at_every_horizon(experiment):
    for report in experiment.reports:
        for segment in (report.validation, report.test):
            assert segment.n_origins > 0
            assert set(segment.metrics_by_horizon) == {1, 2, 3}
            assert segment.metrics_by_horizon[1].mae is not None


def test_validation_window_precedes_the_test_window(experiment):
    for report in experiment.reports:
        assert report.validation.period_end <= report.test.period_start


def test_baseline_ratio_is_one_for_the_baseline_itself(experiment):
    naive = experiment.report_for("naive")
    assert naive.mae_vs_naive_test == pytest.approx(1.0)
    assert naive.beats_naive_on_test is None


def test_model_is_compared_against_the_baseline(experiment):
    linear = experiment.report_for("linear_regression")
    assert linear.mae_vs_naive_test is not None
    assert linear.beats_naive_on_test == (linear.mae_vs_naive_test < 1.0)


def test_leaderboard_is_sorted_by_validation_error(experiment):
    rows = leaderboard_rows(experiment)
    maes = [row["validation_mae"] for row in rows]
    assert maes == sorted(maes)
    assert sum(row["is_best"] for row in rows) == 1


def test_selection_uses_validation_only_and_ignores_the_test_window():
    """A model that looks good on the test window must not win on that basis."""
    from app.ml.metrics import MetricSet
    from app.ml.pipeline import SegmentEvaluation

    def segment(name: str, mae: float) -> SegmentEvaluation:
        return SegmentEvaluation(
            segment=name,
            horizon=1,
            n_origins=10,
            period_start=None,
            period_end=None,
            metrics_by_horizon={1: MetricSet(10, mae, mae, None, None, None)},
            residual_quantiles={},
        )

    good_validation = ModelReport("a", "A", "f", {}, segment("validation", 1.0), segment("test", 9.0))
    good_test = ModelReport("b", "B", "f", {}, segment("validation", 2.0), segment("test", 0.1))
    best, rule = select_best_model([good_validation, good_test])
    assert best == "a"
    assert "test window is never used" in rule


def test_a_failing_model_does_not_abort_the_experiment(monkeypatch, feature_frame):
    """NFR-03: one broken algorithm is reported, the others still run."""

    class BrokenForecaster(BaseForecaster):
        name = "broken"
        display_name = "Broken"
        family = "baseline"
        min_train_rows = 10

        def _fit(self, frame):
            raise RuntimeError("deliberate failure")

        def _predict(self, frame):
            return np.zeros(self.horizon)

    monkeypatch.setitem(MODEL_REGISTRY, "broken", BrokenForecaster)

    from datetime import datetime, timezone

    from app.data.providers import PriceHistory
    from tests.conftest import make_price_frame

    history = PriceHistory("TEST", make_price_frame(), "csv", datetime.now(timezone.utc))
    result = run_experiment(history, horizon=1, model_names=["broken", "naive"])
    broken = result.report_for("broken")
    assert broken is not None and not broken.ok
    assert "deliberate failure" in broken.error
    assert result.report_for("naive").ok
    assert result.best_model == "naive"


def test_walk_forward_never_shows_a_model_a_future_session(monkeypatch, feature_frame):
    """The leakage guarantee, asserted from inside the harness.

    A spy model records the last index of every frame it is handed, so the
    assertion covers both fitting and prediction at each forecast origin.
    """
    seen: list[pd.Timestamp] = []

    class SpyForecaster(BaseForecaster):
        name = "spy"
        display_name = "Spy"
        family = "baseline"
        min_train_rows = 10

        def _fit(self, frame):
            seen.append(frame.index[-1])

        def _predict(self, frame):
            seen.append(frame.index[-1])
            return np.zeros(self.horizon)

    monkeypatch.setitem(MODEL_REGISTRY, "spy", SpyForecaster)

    split = chronological_split(len(feature_frame), 0.7, 0.15)
    evaluation = walk_forward_evaluate(
        "spy", feature_frame, split.test, horizon=2, segment_name="test", origin_step=7
    )
    assert evaluation.n_origins > 0
    assert seen
    # Every frame handed to the model ends at or before the latest origin, and
    # the first origin is the first test row.
    last_origin_date = pd.Timestamp(evaluation.curve[-1]["origin_date"])
    assert max(seen) <= last_origin_date
    assert pd.Timestamp(evaluation.curve[0]["origin_date"]) == feature_frame.index[split.test[0]]


def test_expensive_models_are_fitted_once_per_segment(monkeypatch, feature_frame):
    fits: list[int] = []

    class ExpensiveForecaster(BaseForecaster):
        name = "expensive"
        display_name = "Expensive"
        family = "deep_learning"
        refit_cost = "high"
        min_train_rows = 10

        def _fit(self, frame):
            fits.append(len(frame))

        def _predict(self, frame):
            return np.zeros(self.horizon)

    monkeypatch.setitem(MODEL_REGISTRY, "expensive", ExpensiveForecaster)
    split = chronological_split(len(feature_frame), 0.7, 0.15)
    evaluation = walk_forward_evaluate(
        "expensive", feature_frame, split.test, horizon=1, segment_name="test", origin_step=5
    )
    assert evaluation.refits == 1
    assert len(fits) == 1


def test_cheap_models_refit_on_the_expanding_window(monkeypatch, feature_frame):
    sizes: list[int] = []

    class CheapForecaster(BaseForecaster):
        name = "cheap"
        display_name = "Cheap"
        family = "baseline"
        refit_cost = "low"
        min_train_rows = 10

        def _fit(self, frame):
            sizes.append(len(frame))

        def _predict(self, frame):
            return np.zeros(self.horizon)

    monkeypatch.setitem(MODEL_REGISTRY, "cheap", CheapForecaster)
    split = chronological_split(len(feature_frame), 0.7, 0.15)
    evaluation = walk_forward_evaluate(
        "cheap", feature_frame, split.validation, horizon=1, segment_name="validation", origin_step=5
    )
    assert evaluation.refits == len(sizes) > 1
    assert sizes == sorted(sizes), "each refit must use strictly more history"


def test_horizon_is_capped_by_configuration(experiment):
    from datetime import datetime, timezone

    from app.core.config import settings
    from app.data.providers import PriceHistory
    from tests.conftest import make_price_frame

    history = PriceHistory("TEST", make_price_frame(), "csv", datetime.now(timezone.utc))
    result = run_experiment(history, horizon=settings.max_horizon + 10, model_names=["naive"])
    assert result.horizon == settings.max_horizon


def test_experiment_payload_is_json_serialisable(experiment):
    import json

    payload = json.dumps(experiment.as_dict())
    assert "disclaimer" in payload
    assert pipeline.DISCLAIMER in payload
