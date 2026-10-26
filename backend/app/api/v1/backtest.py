"""Historical strategy backtesting (FR-13, SRS 10.5)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.data.providers import DataUnavailableError, InvalidTickerError, load_history
from app.ml.backtest import run_backtest
from app.ml.pipeline import InsufficientHistoryError
from app.ml.registry import UnknownModelError
from app.schemas.api import BacktestRequest
from app.services.rate_limit import enforce_rate_limit

router = APIRouter(prefix="/backtest", tags=["backtest"])


@router.post(
    "",
    dependencies=[Depends(enforce_rate_limit)],
    summary="Simulate a long/flat rule driven by one model",
)
def backtest(payload: BacktestRequest) -> dict:
    """Walk the test window one session at a time and trade on each forecast.

    The result is a simulation of one decision rule on one historical window,
    reported next to buy-and-hold. It is not a measure of forecast accuracy and
    not a trading recommendation.
    """
    try:
        history = load_history(payload.ticker, days=payload.history_days)
        result = run_backtest(
            history,
            model_name=payload.model,
            threshold_percent=payload.threshold_percent,
            transaction_cost_bps=payload.transaction_cost_bps,
            refit_every=payload.refit_every,
            use_bulk_deals=payload.use_bulk_deals,
        )
    except (InvalidTickerError, UnknownModelError, InsufficientHistoryError, ValueError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except DataUnavailableError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    return result.as_dict()
