"""Model catalogue, comparison and leaderboard (FR-06, SRS 10.2)."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data.providers import DataUnavailableError, InvalidTickerError
from app.db.models import ModelEvaluation, ModelVersion
from app.db.session import get_db
from app.ml.pipeline import InsufficientHistoryError
from app.ml.registry import DEFAULT_MODEL_SET, UnknownModelError, describe_models
from app.schemas.api import CompareRequest, ComparisonResponse, ModelInfoOut
from app.services.forecast_service import (
    comparison_payload,
    get_experiment,
    persist_comparison,
)
from app.services.rate_limit import enforce_rate_limit

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelInfoOut], summary="Available algorithms")
def catalogue() -> list[ModelInfoOut]:
    return [ModelInfoOut(**entry) for entry in describe_models()]


@router.post(
    "/compare",
    response_model=ComparisonResponse,
    dependencies=[Depends(enforce_rate_limit)],
    summary="Evaluate several models on identical windows",
)
def compare(payload: CompareRequest, db: Session = Depends(get_db)) -> ComparisonResponse:
    """Run every requested model on the same splits, origins and target.

    This endpoint trains models and is the slowest in the API; it is rate
    limited and its results are cached per ticker, horizon and trading session.
    """
    models = payload.models or DEFAULT_MODEL_SET
    try:
        result, history = get_experiment(
            payload.ticker,
            horizon=payload.horizon,
            model_names=models,
            history_days=payload.history_days,
            use_bulk_deals=payload.use_bulk_deals,
        )
    except (InvalidTickerError, UnknownModelError, InsufficientHistoryError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except DataUnavailableError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    try:
        persist_comparison(db, result)
    except Exception:
        db.rollback()
        logger.exception("Could not persist comparison for %s", result.ticker)

    return ComparisonResponse(**comparison_payload(result, history))


@router.get("/leaderboard", summary="Stored model rankings")
def leaderboard(
    ticker: str | None = Query(None, max_length=20),
    segment: str = Query("test", pattern="^(validation|test)$"),
    horizon: int = Query(1, ge=1, le=30),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> dict:
    """Most recent stored evaluation per model, ranked by MAE.

    Rankings come from evaluations that were already measured and saved, so the
    table reflects comparable windows rather than a fresh training run.
    """
    query = (
        select(ModelEvaluation, ModelVersion)
        .join(ModelVersion, ModelEvaluation.model_version_id == ModelVersion.id, isouter=True)
        .where(ModelEvaluation.segment == segment, ModelEvaluation.horizon == horizon)
        .order_by(ModelEvaluation.id.desc())
    )
    if ticker:
        query = query.where(ModelEvaluation.ticker == ticker.strip().upper())

    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for evaluation, version in db.execute(query.limit(limit * 4)).all():
        key = (evaluation.ticker, evaluation.model_name)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "model": evaluation.model_name,
                "ticker": evaluation.ticker,
                "segment": evaluation.segment,
                "horizon": evaluation.horizon,
                "period_start": evaluation.period_start.isoformat() if evaluation.period_start else None,
                "period_end": evaluation.period_end.isoformat() if evaluation.period_end else None,
                "n_observations": evaluation.n_observations,
                "mae": evaluation.mae,
                "rmse": evaluation.rmse,
                "mape": evaluation.mape,
                "directional_accuracy": evaluation.directional_accuracy,
                "r2": evaluation.r2,
                "mae_vs_naive": evaluation.mae_vs_naive,
                "training_seconds": version.training_seconds if version else None,
                "evaluated_at": evaluation.evaluated_at.isoformat() if evaluation.evaluated_at else None,
            }
        )
        if len(rows) >= limit:
            break

    rows.sort(key=lambda row: (row["mae"] is None, row["mae"] or 0.0))
    return {
        "segment": segment,
        "horizon": horizon,
        "ticker": ticker.strip().upper() if ticker else None,
        "rows": rows,
        "note": (
            "Rankings are only comparable within the same ticker, horizon and evaluation period. "
            "A model is not 'most accurate' until it beats the naive baseline on unseen data."
        ),
    }
