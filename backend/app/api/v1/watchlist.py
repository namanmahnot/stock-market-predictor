"""Per-user watchlist (FR-08). Every row is scoped to the signed-in user."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.data.providers import DataUnavailableError, InvalidTickerError, load_history
from app.db.models import User, WatchlistItem
from app.db.session import get_db
from app.schemas.api import WatchlistCreate, WatchlistItemOut, WatchlistQuoteOut

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/watchlist", tags=["watchlist"])


@router.get("", response_model=list[WatchlistQuoteOut], summary="Watchlist with latest prices")
def list_watchlist(
    with_quotes: bool = Query(True, description="Attach the latest cached close for each entry."),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[WatchlistQuoteOut]:
    items = db.scalars(
        select(WatchlistItem)
        .where(WatchlistItem.user_id == user.id)
        .order_by(WatchlistItem.created_at.desc())
    ).all()

    output: list[WatchlistQuoteOut] = []
    for item in items:
        entry = WatchlistQuoteOut(id=item.id, ticker=item.ticker, note=item.note)
        if with_quotes:
            try:
                history = load_history(item.ticker, days=90)
                closes = history.frame["adj_close"]
                entry.company_name = history.name
                entry.last_close = float(closes.iloc[-1])
                entry.last_date = history.frame.index[-1].date()
                if len(closes) > 1:
                    entry.change_percent = (float(closes.iloc[-1]) / float(closes.iloc[-2]) - 1.0) * 100.0
            except (DataUnavailableError, InvalidTickerError) as exc:
                # One unavailable symbol must not break the whole dashboard.
                entry.error = str(exc)
        output.append(entry)
    return output


@router.post("", response_model=WatchlistItemOut, status_code=status.HTTP_201_CREATED)
def add_to_watchlist(
    payload: WatchlistCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> WatchlistItemOut:
    existing = db.scalar(
        select(WatchlistItem).where(
            WatchlistItem.user_id == user.id, WatchlistItem.ticker == payload.ticker
        )
    )
    if existing is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"{payload.ticker} is already on your watchlist."
        )
    item = WatchlistItem(user_id=user.id, ticker=payload.ticker, note=payload.note)
    db.add(item)
    db.commit()
    db.refresh(item)
    return WatchlistItemOut.model_validate(item)


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_from_watchlist(
    item_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    item = db.get(WatchlistItem, item_id)
    # A row belonging to someone else is reported as missing, not as forbidden.
    if item is None or item.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No watchlist entry with id {item_id}.")
    db.delete(item)
    db.commit()
