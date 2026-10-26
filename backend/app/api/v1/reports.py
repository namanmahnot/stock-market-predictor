"""Report downloads (FR-09)."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_optional_user
from app.data.providers import DataUnavailableError, InvalidTickerError, load_history
from app.db.models import ModelEvaluation, PredictionRun, User
from app.db.session import get_db
from app.ml.pipeline import InsufficientHistoryError
from app.ml.registry import DEFAULT_MODEL_SET, UnknownModelError
from app.reports.builder import comparison_csv, forecast_csv, forecast_pdf
from app.schemas.api import CompareRequest
from app.services.forecast_service import comparison_payload, get_experiment
from app.services.rate_limit import enforce_rate_limit

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/reports", tags=["reports"])


def _load_run(run_id: int, db: Session, user: User | None) -> PredictionRun:
    run = db.get(PredictionRun, run_id)
    if run is None or (run.user_id is not None and (user is None or run.user_id != user.id)):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No prediction run with id {run_id}.")
    return run


@router.get("/{run_id}", summary="Download a saved prediction as CSV or PDF")
def download_report(
    run_id: int,
    format: str = Query("csv", pattern="^(csv|pdf)$"),
    db: Session = Depends(get_db),
    user: User | None = Depends(get_optional_user),
) -> Response:
    run = _load_run(run_id, db, user)
    evaluations = db.scalars(
        select(ModelEvaluation)
        .where(ModelEvaluation.run_id == run.id)
        .order_by(ModelEvaluation.segment, ModelEvaluation.horizon)
    ).all()

    if format == "csv":
        body = forecast_csv(run, evaluations)
        return Response(
            content=body,
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{run.ticker}_run{run.id}_forecast.csv"'
            },
        )

    history = None
    try:
        history = load_history(run.ticker, days=400).frame
    except (DataUnavailableError, InvalidTickerError):
        # The PDF still carries the forecast and metrics without the chart.
        logger.warning("No history available for the report chart of %s", run.ticker)

    pdf = forecast_pdf(run, evaluations, history)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{run.ticker}_run{run.id}_report.pdf"'
        },
    )


@router.post(
    "/comparison",
    dependencies=[Depends(enforce_rate_limit)],
    summary="Download a model comparison as CSV",
)
def download_comparison(payload: CompareRequest) -> Response:
    try:
        result, history = get_experiment(
            payload.ticker,
            horizon=payload.horizon,
            model_names=payload.models or DEFAULT_MODEL_SET,
            history_days=payload.history_days,
            use_bulk_deals=payload.use_bulk_deals,
        )
    except (InvalidTickerError, UnknownModelError, InsufficientHistoryError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except DataUnavailableError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    payload_dict = comparison_payload(result, history)
    payload_dict["reference_date"] = result.reference_date.isoformat()
    body = comparison_csv(payload_dict)
    return Response(
        content=body,
        media_type="text/csv",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{result.ticker}_model_comparison_h{result.horizon}.csv"'
            )
        },
    )
