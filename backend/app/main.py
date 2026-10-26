"""FastAPI application entry point.

Run from the backend directory with:
    uvicorn app.main:app --reload
Interactive API documentation is served at /docs.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import settings
from app.data.providers import DataUnavailableError, InvalidTickerError
from app.db.init_db import init_db
from app.ml.pipeline import InsufficientHistoryError
from app.ml.registry import UnknownModelError
from app.schemas.api import HealthOut

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

DESCRIPTION = """
AI-powered historical stock analysis and price forecasting for an academic
laboratory project.

* Historical OHLCV retrieval with the data source and retrieval time attached
  to every response.
* Technical indicators computed only from sessions available at each row.
* Six comparable forecasting algorithms, from a naive baseline to an LSTM.
* Walk-forward evaluation on chronological train / validation / test windows,
  always reported against the naive baseline.
* Watchlists, saved prediction runs, SHAP explanations, strategy backtesting
  and CSV / PDF report export.

Forecasts are statistical estimates for educational analysis. They are not
investment advice and no future price is guaranteed.
"""


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    init_db()
    logger.info("%s started in %s mode", settings.app_name, settings.environment)
    yield


app = FastAPI(
    lifespan=lifespan,
    title=settings.app_name,
    description=DESCRIPTION,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.exception_handler(InvalidTickerError)
def handle_invalid_ticker(request: Request, exc: InvalidTickerError) -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": str(exc)})


@app.exception_handler(UnknownModelError)
def handle_unknown_model(request: Request, exc: UnknownModelError) -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": str(exc)})


@app.exception_handler(InsufficientHistoryError)
def handle_insufficient_history(request: Request, exc: InsufficientHistoryError) -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": str(exc)})


@app.exception_handler(DataUnavailableError)
def handle_data_unavailable(request: Request, exc: DataUnavailableError) -> JSONResponse:
    """A provider outage is reported as such; prices are never invented."""
    return JSONResponse(status_code=status.HTTP_502_BAD_GATEWAY, content={"detail": str(exc)})


@app.get("/api/health", response_model=HealthOut, tags=["system"])
def health() -> HealthOut:
    backend = "postgresql" if settings.database_url.startswith("postgresql") else "sqlite"
    return HealthOut(
        status="ok",
        app_name=settings.app_name,
        environment=settings.environment,
        data_provider=settings.data_provider,
        database=backend,
    )


app.include_router(api_router, prefix=settings.api_v1_prefix)
