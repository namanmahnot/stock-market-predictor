"""Orchestration between the API layer, the ML pipeline and the database.

Experiment results are memoised per (ticker, horizon, model set, history
window, latest session) so that repeated dashboard interactions during a
demonstration do not retrain anything, while a new trading session or a
different request still triggers a fresh run.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import date
from typing import Any, Iterable

from sqlalchemy.orm import Session

from app.core.config import settings
from app.data.providers import PriceHistory, load_history
from app.db.models import (
    ModelEvaluation,
    ModelVersion,
    PredictionResult,
    PredictionRun,
    User,
)
from app.ml.pipeline import (
    DISCLAIMER,
    ExperimentResult,
    ModelReport,
    leaderboard_rows,
    run_experiment,
)
from app.ml.registry import BASELINE_MODEL, DEFAULT_MODEL_SET

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 15 * 60
_cache: dict[tuple, tuple[float, ExperimentResult]] = {}
_cache_lock = threading.Lock()

LIMITATIONS = [
    "Forecasts are extrapolations from historical price behaviour; they cannot anticipate news, earnings surprises or macroeconomic shocks.",
    "Metrics are measured on one ticker over one historical window and will differ in other market regimes.",
    "Target dates are estimated from weekday sessions and do not account for exchange holidays.",
    "Prediction intervals come from the empirical spread of validation residuals and assume that spread continues to hold.",
]


def _cache_key(
    ticker: str,
    horizon: int,
    models: tuple[str, ...],
    history_days: int,
    last_date: date,
    use_bulk_deals: bool | None,
) -> tuple:
    # The feature set is part of the key: the same models on the same sessions
    # produce different numbers with and without the bulk-deal group.
    return (ticker, horizon, models, history_days, last_date.isoformat(), use_bulk_deals)


def get_experiment(
    ticker: str,
    horizon: int,
    model_names: Iterable[str] | None = None,
    history_days: int | None = None,
    use_bulk_deals: bool | None = None,
    use_cache: bool = True,
) -> tuple[ExperimentResult, PriceHistory]:
    """Run (or reuse) an experiment for one ticker and model set."""
    history_days = int(history_days or settings.default_history_days)
    history = load_history(ticker, days=history_days)
    requested = tuple(sorted(set(model_names or DEFAULT_MODEL_SET) | {BASELINE_MODEL}))
    key = _cache_key(
        history.ticker, horizon, requested, history_days, history.last_date.date(), use_bulk_deals
    )

    if use_cache:
        with _cache_lock:
            entry = _cache.get(key)
        if entry and time.time() - entry[0] < _CACHE_TTL_SECONDS:
            logger.debug("Reusing cached experiment for %s", key)
            return entry[1], history

    result = run_experiment(
        history,
        horizon=horizon,
        model_names=list(requested),
        history_days=history_days,
        use_bulk_deals=use_bulk_deals,
        save_artifacts=True,
    )
    with _cache_lock:
        _cache[key] = (time.time(), result)
        if len(_cache) > 32:
            oldest = min(_cache.items(), key=lambda item: item[1][0])[0]
            _cache.pop(oldest, None)
    return result, history


def clear_experiment_cache() -> None:
    with _cache_lock:
        _cache.clear()


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
def persist_run(
    db: Session,
    user: User | None,
    result: ExperimentResult,
    report: ModelReport,
    auto_selected: bool,
) -> PredictionRun:
    """Store a prediction run, its forecasts and its evaluation metrics."""
    version = ModelVersion(
        model_name=report.name,
        version=f"h{result.horizon}-{result.feature_set}-{result.reference_date.isoformat()}",
        ticker=result.ticker,
        target="cumulative_log_return",
        horizon=result.horizon,
        features=json.dumps(result.feature_columns),
        hyperparameters=json.dumps(report.hyperparameters, default=str),
        train_start=date.fromisoformat(result.split_dates["train_start"]),
        train_end=result.reference_date,
        artifact_path=report.artifact_path,
        training_seconds=report.training_seconds,
    )
    db.add(version)
    db.flush()

    run = PredictionRun(
        user_id=user.id if user else None,
        ticker=result.ticker,
        model_name=report.name,
        model_version_id=version.id,
        forecast_horizon=result.horizon,
        history_days=result.history_days,
        reference_date=result.reference_date,
        reference_price=result.reference_price,
        data_source=result.data_source,
        data_retrieved_at=result.data_retrieved_at,
        auto_selected=auto_selected,
        status="completed" if report.ok else "failed",
        error_message=report.error,
        runtime_seconds=result.runtime_seconds,
    )
    db.add(run)
    db.flush()

    for step in report.forecast:
        db.add(
            PredictionResult(
                run_id=run.id,
                step=step.step,
                target_date=step.target_date,
                predicted_price=step.predicted_price,
                lower_bound=step.lower_bound,
                upper_bound=step.upper_bound,
            )
        )

    for segment in (report.validation, report.test):
        if segment is None:
            continue
        for horizon, metrics in segment.metrics_by_horizon.items():
            db.add(
                ModelEvaluation(
                    model_version_id=version.id,
                    run_id=run.id,
                    model_name=report.name,
                    ticker=result.ticker,
                    segment=segment.segment,
                    horizon=horizon,
                    period_start=segment.period_start,
                    period_end=segment.period_end,
                    n_observations=metrics.n_observations,
                    mae=metrics.mae,
                    rmse=metrics.rmse,
                    mape=metrics.mape,
                    directional_accuracy=metrics.directional_accuracy,
                    r2=metrics.r2,
                    mae_vs_naive=(
                        report.mae_vs_naive_validation
                        if segment.segment == "validation"
                        else report.mae_vs_naive_test
                    )
                    if horizon == 1
                    else None,
                )
            )

    db.commit()
    db.refresh(run)
    return run


# ---------------------------------------------------------------------------
# response assembly
# ---------------------------------------------------------------------------
def provenance_payload(history: PriceHistory) -> dict[str, Any]:
    return {
        "source": history.source,
        "retrieved_at": history.retrieved_at,
        "from_cache": history.from_cache,
        "warnings": history.warnings,
    }


def prediction_payload(
    result: ExperimentResult,
    report: ModelReport,
    history: PriceHistory,
    run_id: int | None,
    auto_selected: bool,
    explanation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    baseline = result.report_for(BASELINE_MODEL)
    test = report.test.primary if report.test else None
    validation = report.validation.primary if report.validation else None
    metrics = test or validation
    metrics_source = (
        "Walk-forward on the held-out test window"
        if test
        else "Walk-forward on the validation window"
        if validation
        else None
    )
    by_horizon = report.test or report.validation

    return {
        "run_id": run_id,
        "ticker": result.ticker,
        "company_name": result.company_name,
        "currency": result.currency,
        "model": report.name,
        "model_display_name": report.display_name,
        "auto_selected": auto_selected,
        "selection_rule": result.selection_rule if auto_selected else None,
        "forecast_horizon": result.horizon,
        "reference_date": result.reference_date,
        "reference_price": result.reference_price,
        "forecasts": [step.as_dict() for step in report.forecast],
        "metrics": metrics.as_dict() if metrics else None,
        "metrics_source": metrics_source,
        "metrics_by_horizon": (
            {str(h): m.as_dict() for h, m in sorted(by_horizon.metrics_by_horizon.items())}
            if by_horizon
            else {}
        ),
        "mae_vs_naive_test": report.mae_vs_naive_test,
        "beats_naive_on_test": report.beats_naive_on_test,
        "baseline_metrics": (
            baseline.test.primary.as_dict()
            if baseline and baseline.ok and baseline.test and baseline.test.primary
            else None
        ),
        "dataset": {
            "price_rows": result.n_rows,
            "feature_rows": result.feature_rows,
            "feature_set": result.feature_set,
            "n_features": len(result.feature_columns),
            "notes": result.dataset_notes,
            **result.split.describe(),
            **result.split_dates,
        },
        "feature_set": result.feature_set,
        "bulk_deals": result.bulk_deals,
        "provenance": provenance_payload(history),
        "explanation": explanation,
        "limitations": LIMITATIONS,
        "runtime_seconds": result.runtime_seconds,
        "disclaimer": DISCLAIMER,
    }


def comparison_payload(result: ExperimentResult, history: PriceHistory) -> dict[str, Any]:
    return {
        "ticker": result.ticker,
        "forecast_horizon": result.horizon,
        "reference_date": result.reference_date,
        "reference_price": result.reference_price,
        "best_model": result.best_model,
        "selection_rule": result.selection_rule,
        "dataset": {
            "price_rows": result.n_rows,
            "feature_rows": result.feature_rows,
            "feature_set": result.feature_set,
            "n_features": len(result.feature_columns),
            "notes": result.dataset_notes,
            **result.split.describe(),
            **result.split_dates,
        },
        "feature_set": result.feature_set,
        "bulk_deals": result.bulk_deals,
        "provenance": provenance_payload(history),
        "leaderboard": leaderboard_rows(result),
        "models": [report.as_dict(include_curve=True) for report in result.reports],
        "runtime_seconds": result.runtime_seconds,
        "disclaimer": DISCLAIMER,
    }


def persist_comparison(db: Session, result: ExperimentResult) -> int:
    """Store model versions and evaluation rows for a comparison run.

    These rows back the leaderboard endpoint, so a demonstration can show
    rankings that were measured earlier instead of retraining on every visit.
    """
    stored = 0
    for report in result.reports:
        if not report.ok:
            continue
        version = ModelVersion(
            model_name=report.name,
            version=f"h{result.horizon}-{result.feature_set}-{result.reference_date.isoformat()}",
            ticker=result.ticker,
            target="cumulative_log_return",
            horizon=result.horizon,
            features=json.dumps(result.feature_columns),
            hyperparameters=json.dumps(report.hyperparameters, default=str),
            train_start=date.fromisoformat(result.split_dates["train_start"]),
            train_end=result.reference_date,
            artifact_path=report.artifact_path,
            training_seconds=report.training_seconds,
        )
        db.add(version)
        db.flush()
        for segment in (report.validation, report.test):
            if segment is None:
                continue
            metrics = segment.metrics_by_horizon.get(1)
            if metrics is None:
                continue
            db.add(
                ModelEvaluation(
                    model_version_id=version.id,
                    model_name=report.name,
                    ticker=result.ticker,
                    segment=segment.segment,
                    horizon=1,
                    period_start=segment.period_start,
                    period_end=segment.period_end,
                    n_observations=metrics.n_observations,
                    mae=metrics.mae,
                    rmse=metrics.rmse,
                    mape=metrics.mape,
                    directional_accuracy=metrics.directional_accuracy,
                    r2=metrics.r2,
                    mae_vs_naive=(
                        report.mae_vs_naive_validation
                        if segment.segment == "validation"
                        else report.mae_vs_naive_test
                    ),
                )
            )
            stored += 1
    db.commit()
    return stored
