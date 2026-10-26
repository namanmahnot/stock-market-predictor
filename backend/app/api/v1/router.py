"""Aggregates every version 1 route under a single prefix."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import (
    auth,
    backtest,
    bulk_deals,
    models,
    predictions,
    reports,
    stocks,
    watchlist,
)

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(stocks.router)
api_router.include_router(bulk_deals.router)
api_router.include_router(predictions.router)
api_router.include_router(models.router)
api_router.include_router(watchlist.router)
api_router.include_router(backtest.router)
api_router.include_router(reports.router)
