"""NSE bulk-deal disclosures (FR-15).

Two uses: an analytical view of who traded a stock in size, and the optional
feature group the forecasting models can be given.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from app.core.config import settings
from app.data.bulk_deals import (
    BulkDealsUnavailableError,
    load_bulk_deals,
    symbol_for_ticker,
)
from app.data.providers import SYMBOL_UNIVERSE, InvalidTickerError, normalise_ticker
from app.schemas.api import BulkDealsOut, BulkDealSymbolOut

router = APIRouter(prefix="/bulk-deals", tags=["bulk deals"])


@router.get("/symbols", response_model=list[BulkDealSymbolOut], summary="Symbols in the dataset")
def symbols(
    limit: int = Query(50, ge=1, le=500),
    only_sufficient: bool = Query(
        False, description="Keep only symbols with enough disclosed sessions to model."
    ),
) -> list[BulkDealSymbolOut]:
    """Symbols ranked by how many sessions carry a disclosed deal.

    Useful because most tickers appear on only one or two sessions, and the
    large caps elsewhere in the demo appear on none.
    """
    try:
        dataset = load_bulk_deals()
    except BulkDealsUnavailableError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc

    rows = dataset.coverage_table()
    if only_sufficient:
        rows = [row for row in rows if row["sufficient"]]
    return [BulkDealSymbolOut(**row) for row in rows[:limit]]


@router.get("/{ticker}", response_model=BulkDealsOut, summary="Disclosed deals for one ticker")
def deals_for_ticker(
    ticker: str,
    limit: int = Query(100, ge=1, le=1000, description="Most recent disclosures to return."),
) -> BulkDealsOut:
    try:
        ticker = normalise_ticker(ticker)
    except InvalidTickerError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    try:
        dataset = load_bulk_deals()
    except BulkDealsUnavailableError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc

    coverage = dataset.coverage(ticker)
    notes: list[str] = []
    known = next((entry for entry in SYMBOL_UNIVERSE if entry["ticker"] == ticker), None)
    non_nse = symbol_for_ticker(ticker) is None or (known and known["exchange"] != "NSE")
    if non_nse:
        notes.append(
            "This report covers NSE equities only, so a non-NSE symbol can never appear in it."
        )
    elif not coverage.in_dataset:
        notes.append(
            f"{ticker} has no disclosed bulk deals between {coverage.dataset_start} and "
            f"{coverage.dataset_end}. A bulk deal requires one client to trade more than 0.5% "
            "of the company's equity in a single session, which rarely happens for large caps."
        )
    elif not coverage.sufficient:
        notes.append(
            f"{coverage.sessions_with_deals} disclosed session(s) is below the minimum of "
            f"{settings.bulk_deals_min_sessions} used before these features are given to a model; "
            "the records below are still shown for analysis."
        )

    rows = dataset.for_ticker(ticker).sort_values("deal_date", ascending=False).head(limit)
    daily = dataset.daily(ticker).sort_index(ascending=False)

    return BulkDealsOut(
        ticker=ticker,
        coverage=coverage.as_dict(),
        sources=list(dataset.sources),
        deals=[
            {
                "deal_date": row.deal_date.date(),
                "security_name": row.security_name,
                "client_name": row.client_name,
                "side": row.side,
                "quantity": float(row.quantity),
                "price": float(row.price),
                "turnover": float(row.turnover),
            }
            for row in rows.itertuples()
        ],
        daily=[
            {
                "date": index.date(),
                "deal_count": float(row.deal_count),
                "buy_quantity": float(row.buy_quantity),
                "sell_quantity": float(row.sell_quantity),
                "net_quantity": float(row.net_quantity),
                "gross_turnover": float(row.gross_turnover),
                "deal_vwap": None if row.deal_vwap != row.deal_vwap else float(row.deal_vwap),
            }
            for index, row in daily.iterrows()
        ],
        notes=notes,
    )
