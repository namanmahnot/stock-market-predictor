"""Forecast generation and prediction history (FR-05, FR-08, FR-10, FR-12)."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_optional_user
from app.data.providers import DataUnavailableError, InvalidTickerError
from app.db.models import ModelEvaluation, PredictionRun, User
from app.db.session import get_db
from app.ml.pipeline import InsufficientHistoryError, prepare_dataset
from app.ml.registry import UnknownModelError, create_model
from app.schemas.api import (
    ForecastStepOut,
    MetricsOut,
    PredictionRequest,
    PredictionResponse,
    PredictionRunOut,
)
from app.services.forecast_service import (
    get_experiment,
    persist_run,
    prediction_payload,
)
from app.services.rate_limit import enforce_rate_limit

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post(
    "",
    response_model=PredictionResponse,
    dependencies=[Depends(enforce_rate_limit)],
    summary="Generate a forecast for one ticker",
)
def create_prediction(
    payload: PredictionRequest,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_optional_user),
) -> PredictionResponse:
    """Train or reuse the requested model and return its forecast.

    With ``model: "auto"`` the model with the lowest validation MAE is chosen;
    the held-out test window is never used to make that choice.
    """
    auto = payload.model == "auto"
    # "auto" has to evaluate the whole default set; a named model is compared
    # against the naive baseline only, which keeps the request fast.
    model_names = None if auto else [payload.model]

    try:
        result, history = get_experiment(
            payload.ticker,
            horizon=payload.horizon,
            model_names=model_names,
            history_days=payload.history_days,
            use_bulk_deals=payload.use_bulk_deals,
        )
    except (InvalidTickerError, UnknownModelError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except InsufficientHistoryError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except DataUnavailableError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    chosen = result.best_model if auto else payload.model
    report = result.report_for(chosen)
    if report is None:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "No report was produced.")
    if not report.ok:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, report.error or "Model failed.")

    explanation = None
    if payload.include_explanation:
        explanation = _explain(report.name, history, result.horizon, payload.use_bulk_deals)

    run_id = None
    if payload.save:
        try:
            run = persist_run(db, user, result, report, auto_selected=auto)
            run_id = run.id
        except Exception:  # a storage problem must not discard a valid forecast
            db.rollback()
            logger.exception("Could not persist run for %s/%s", result.ticker, report.name)

    payload_dict = prediction_payload(result, report, history, run_id, auto, explanation)
    return PredictionResponse(**payload_dict)


def _explain(model_name: str, history, horizon: int, use_bulk_deals: bool | None) -> dict | None:
    """Refit the model on all history and explain its latest prediction.

    The same feature set is rebuilt here, so the contributions describe the
    model that produced the forecast rather than a differently-fed twin.
    """
    try:
        model = create_model(model_name, horizon=horizon)
        if not model.supports_explanations:
            return {
                "unavailable": f"{model.display_name} does not expose SHAP contributions.",
            }
        frame = prepare_dataset(history, use_bulk_deals=use_bulk_deals).frame
        model.fit(frame)
        return model.explain(frame)
    except Exception:
        logger.warning("Explanation failed for %s", model_name, exc_info=True)
        return None


@router.get("", response_model=list[PredictionRunOut], summary="Prediction history")
def list_runs(
    limit: int = Query(25, ge=1, le=100),
    ticker: str | None = Query(None, max_length=20),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[PredictionRunOut]:
    """Saved runs belonging to the authenticated user only (SRS 16.6)."""
    query = select(PredictionRun).where(PredictionRun.user_id == user.id)
    if ticker:
        query = query.where(PredictionRun.ticker == ticker.upper())
    runs = db.scalars(query.order_by(PredictionRun.id.desc()).limit(limit)).all()
    return [_run_out(db, run) for run in runs]


@router.get("/{run_id}", response_model=PredictionRunOut, summary="Retrieve a saved prediction")
def get_run(
    run_id: int,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_optional_user),
) -> PredictionRunOut:
    run = db.get(PredictionRun, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No prediction run with id {run_id}.")
    # A run saved by a signed-in user is private to that user.
    if run.user_id is not None and (user is None or run.user_id != user.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No prediction run with id {run_id}.")
    return _run_out(db, run)


def _run_out(db: Session, run: PredictionRun) -> PredictionRunOut:
    evaluation = db.scalar(
        select(ModelEvaluation)
        .where(
            ModelEvaluation.run_id == run.id,
            ModelEvaluation.segment == "test",
            ModelEvaluation.horizon == 1,
        )
        .limit(1)
    )
    metrics = (
        MetricsOut(
            n_observations=evaluation.n_observations,
            mae=evaluation.mae,
            rmse=evaluation.rmse,
            mape=evaluation.mape,
            directional_accuracy=evaluation.directional_accuracy,
            r2=evaluation.r2,
        )
        if evaluation
        else None
    )
    return PredictionRunOut(
        id=run.id,
        ticker=run.ticker,
        model_name=run.model_name,
        forecast_horizon=run.forecast_horizon,
        reference_date=run.reference_date,
        reference_price=run.reference_price,
        status=run.status,
        auto_selected=run.auto_selected,
        data_source=run.data_source,
        created_at=run.created_at,
        runtime_seconds=run.runtime_seconds,
        forecasts=[
            ForecastStepOut(
                step=item.step,
                target_date=item.target_date,
                predicted_price=item.predicted_price,
                lower_bound=item.lower_bound,
                upper_bound=item.upper_bound,
                change=item.predicted_price - (run.reference_price or item.predicted_price),
                change_percent=(
                    (item.predicted_price / run.reference_price - 1.0) * 100.0
                    if run.reference_price
                    else 0.0
                ),
            )
            for item in run.results
        ],
        metrics=metrics,
        error_message=run.error_message,
    )
