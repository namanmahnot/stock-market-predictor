"""Stock search, historical prices and technical indicators (FR-01 to FR-04)."""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException, Query, status

from app.core.config import settings
from app.data.providers import (
    DataUnavailableError,
    InvalidTickerError,
    load_history,
    search_symbols,
)
from app.ml.features import INDICATOR_COLUMNS, add_indicators
from app.schemas.api import HistoryOut, IndicatorsOut, SymbolOut
from app.services.forecast_service import provenance_payload

router = APIRouter(prefix="/stocks", tags=["stocks"])

INDICATOR_DEFINITIONS = {
    "sma_5": "Simple moving average of the adjusted close over 5 sessions.",
    "sma_10": "Simple moving average over 10 sessions.",
    "sma_20": "Simple moving average over 20 sessions.",
    "sma_50": "Simple moving average over 50 sessions.",
    "ema_12": "Exponential moving average (span 12), weighting recent sessions more heavily.",
    "ema_26": "Exponential moving average (span 26).",
    "rsi_14": "Wilder's Relative Strength Index over 14 sessions, scaled 0-100. Above 70 is often read as overbought, below 30 as oversold.",
    "macd": "Difference between the 12- and 26-session exponential moving averages.",
    "macd_signal": "9-session exponential moving average of the MACD line.",
    "macd_hist": "MACD line minus its signal line.",
    "daily_return": "Percentage change of the adjusted close from the previous session.",
    "volatility_20": "Standard deviation of the last 20 daily log returns.",
    "volume_change": "Percentage change in traded volume from the previous session.",
}


def _handle_data_error(exc: Exception) -> HTTPException:
    if isinstance(exc, InvalidTickerError):
        return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc))
    return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


@router.get("/search", response_model=list[SymbolOut])
def search(
    q: str = Query("", max_length=40, description="Ticker or company name fragment"),
    limit: int = Query(10, ge=1, le=50),
) -> list[SymbolOut]:
    return [SymbolOut(**entry) for entry in search_symbols(q, limit)]


@router.get("/{ticker}/history", response_model=HistoryOut)
def history(
    ticker: str,
    days: int = Query(settings.default_history_days, ge=30, le=settings.max_history_days),
) -> HistoryOut:
    """Historical OHLCV rows with their source and retrieval timestamp (FR-02)."""
    try:
        price_history = load_history(ticker, days=days)
    except (DataUnavailableError, InvalidTickerError) as exc:
        raise _handle_data_error(exc) from exc

    frame = price_history.frame
    closes = frame["adj_close"]
    previous = float(closes.iloc[-2]) if len(closes) > 1 else None
    last = float(closes.iloc[-1])

    bars = [
        {
            "date": index.date(),
            "open": _clean(row["open"]),
            "high": _clean(row["high"]),
            "low": _clean(row["low"]),
            "close": _clean(row["close"]),
            "adj_close": float(row["adj_close"]),
            "volume": _clean(row["volume"]),
        }
        for index, row in frame.iterrows()
    ]

    return HistoryOut(
        ticker=price_history.ticker,
        company_name=price_history.name,
        currency=price_history.currency,
        n_rows=len(frame),
        first_date=frame.index[0].date(),
        last_date=frame.index[-1].date(),
        last_close=last,
        previous_close=previous,
        change=None if previous is None else last - previous,
        change_percent=None if previous is None else (last / previous - 1.0) * 100.0,
        provenance=provenance_payload(price_history),
        bars=bars,
    )


@router.get("/{ticker}/indicators", response_model=IndicatorsOut)
def indicators(
    ticker: str,
    days: int = Query(settings.default_history_days, ge=60, le=settings.max_history_days),
) -> IndicatorsOut:
    """Technical indicators computed only from sessions up to each row (FR-04)."""
    try:
        price_history = load_history(ticker, days=days)
    except (DataUnavailableError, InvalidTickerError) as exc:
        raise _handle_data_error(exc) from exc

    enriched = add_indicators(price_history.frame)
    columns = ["adj_close", *INDICATOR_COLUMNS]
    points = [
        {
            "date": index.date().isoformat(),
            **{column: _clean(row[column]) for column in columns},
        }
        for index, row in enriched.iterrows()
    ]

    return IndicatorsOut(
        ticker=price_history.ticker,
        columns=columns,
        provenance=provenance_payload(price_history),
        points=points,
        definitions=INDICATOR_DEFINITIONS,
    )


def _clean(value) -> float | None:
    """JSON cannot carry NaN; warm-up rows are sent as null instead."""
    number = float(value)
    return None if not np.isfinite(number) else number
