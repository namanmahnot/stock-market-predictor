"""Pydantic request and response models.

Validation lives here so that malformed tickers, unsupported models and
out-of-range horizons are rejected before any data retrieval or model run
(SRS section 16, item 3).
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.core.config import settings
from app.data.providers import InvalidTickerError, normalise_ticker
from app.ml.registry import MODEL_REGISTRY


def _validate_ticker(value: str) -> str:
    try:
        return normalise_ticker(value)
    except InvalidTickerError as exc:
        raise ValueError(str(exc)) from exc


def _validate_model(value: str) -> str:
    key = (value or "").strip().lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{value}'. Supported: {', '.join(sorted(MODEL_REGISTRY))}."
        )
    return key


class HealthOut(BaseModel):
    status: str
    app_name: str
    environment: str
    data_provider: str
    database: str


# -- stocks ----------------------------------------------------------------
class SymbolOut(BaseModel):
    ticker: str
    name: str
    exchange: str
    currency: str


class PriceBarOut(BaseModel):
    date: date
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    adj_close: float
    volume: float | None


class DataProvenance(BaseModel):
    """Source and freshness, shown in the UI next to every number (SRS 5.1)."""

    source: str
    retrieved_at: datetime
    from_cache: bool
    warnings: list[str] = Field(default_factory=list)


class HistoryOut(BaseModel):
    ticker: str
    company_name: str | None = None
    currency: str | None = None
    n_rows: int
    first_date: date
    last_date: date
    last_close: float
    previous_close: float | None = None
    change: float | None = None
    change_percent: float | None = None
    provenance: DataProvenance
    bars: list[PriceBarOut]


class IndicatorPointOut(BaseModel):
    model_config = ConfigDict(extra="allow")

    date: date
    adj_close: float


class IndicatorsOut(BaseModel):
    ticker: str
    columns: list[str]
    provenance: DataProvenance
    points: list[dict[str, Any]]
    definitions: dict[str, str]


# -- models ----------------------------------------------------------------
class ModelInfoOut(BaseModel):
    name: str
    display_name: str
    family: str
    description: str
    available: bool
    unavailable_reason: str | None
    in_default_set: bool
    supports_explanations: bool
    refit_cost: str
    min_train_rows: int


class HorizonRequestMixin(BaseModel):
    ticker: str = Field(..., examples=["AAPL"], max_length=20)
    horizon: int = Field(1, ge=1, le=settings.max_horizon)
    history_days: int = Field(
        settings.default_history_days, ge=settings.min_observations, le=settings.max_history_days
    )

    @field_validator("ticker")
    @classmethod
    def check_ticker(cls, value: str) -> str:
        return _validate_ticker(value)


USE_BULK_DEALS_DESCRIPTION = (
    "NSE bulk-deal feature group: true requires it, false never uses it, and null (the "
    "default) uses it when the ticker has enough disclosed sessions."
)


class PredictionRequest(HorizonRequestMixin):
    model: str = Field("auto", description="Registry model name, or 'auto' to select on validation MAE.")
    use_bulk_deals: bool | None = Field(None, description=USE_BULK_DEALS_DESCRIPTION)
    save: bool = Field(True, description="Persist the run and its forecasts for the prediction history.")
    include_explanation: bool = False

    @field_validator("model")
    @classmethod
    def check_model(cls, value: str) -> str:
        key = (value or "").strip().lower()
        return "auto" if key in {"", "auto", "auto_select"} else _validate_model(key)


class CompareRequest(HorizonRequestMixin):
    models: list[str] = Field(default_factory=list, max_length=len(MODEL_REGISTRY))
    use_bulk_deals: bool | None = Field(None, description=USE_BULK_DEALS_DESCRIPTION)

    @field_validator("models")
    @classmethod
    def check_models(cls, value: list[str]) -> list[str]:
        return [_validate_model(item) for item in value]


class BacktestRequest(BaseModel):
    ticker: str = Field(..., max_length=20)
    model: str = Field("xgboost")
    history_days: int = Field(
        settings.default_history_days, ge=settings.min_observations, le=settings.max_history_days
    )
    threshold_percent: float = Field(0.0, ge=-5.0, le=5.0)
    transaction_cost_bps: float = Field(10.0, ge=0.0, le=200.0)
    refit_every: int = Field(5, ge=1, le=50)
    use_bulk_deals: bool | None = Field(None, description=USE_BULK_DEALS_DESCRIPTION)

    @field_validator("ticker")
    @classmethod
    def check_ticker(cls, value: str) -> str:
        return _validate_ticker(value)

    @field_validator("model")
    @classmethod
    def check_model(cls, value: str) -> str:
        return _validate_model(value)


class ForecastStepOut(BaseModel):
    step: int
    target_date: date
    predicted_price: float
    lower_bound: float | None = None
    upper_bound: float | None = None
    change: float
    change_percent: float


class MetricsOut(BaseModel):
    n_observations: int
    mae: float | None = None
    rmse: float | None = None
    mape: float | None = None
    directional_accuracy: float | None = None
    r2: float | None = None


class PredictionResponse(BaseModel):
    """Response for POST /predictions (FR-05, FR-10)."""

    run_id: int | None = None
    ticker: str
    company_name: str | None = None
    currency: str | None = None
    model: str
    model_display_name: str
    auto_selected: bool
    selection_rule: str | None = None
    forecast_horizon: int
    reference_date: date
    reference_price: float
    forecasts: list[ForecastStepOut]
    metrics: MetricsOut | None = None
    metrics_source: str | None = None
    metrics_by_horizon: dict[str, MetricsOut] = Field(default_factory=dict)
    mae_vs_naive_test: float | None = None
    beats_naive_on_test: bool | None = None
    baseline_metrics: MetricsOut | None = None
    dataset: dict[str, Any] = Field(default_factory=dict)
    feature_set: str | None = None
    bulk_deals: dict[str, Any] | None = None
    provenance: DataProvenance
    explanation: dict[str, Any] | None = None
    limitations: list[str] = Field(default_factory=list)
    runtime_seconds: float | None = None
    disclaimer: str


class ComparisonResponse(BaseModel):
    ticker: str
    forecast_horizon: int
    reference_date: date
    reference_price: float
    best_model: str | None
    selection_rule: str
    dataset: dict[str, Any]
    feature_set: str | None = None
    bulk_deals: dict[str, Any] | None = None
    provenance: DataProvenance
    leaderboard: list[dict[str, Any]]
    models: list[dict[str, Any]]
    runtime_seconds: float
    disclaimer: str


class BulkDealOut(BaseModel):
    deal_date: date
    security_name: str
    client_name: str
    side: Literal["BUY", "SELL"]
    quantity: float
    price: float
    turnover: float


class BulkDealDailyOut(BaseModel):
    date: date
    deal_count: float
    buy_quantity: float
    sell_quantity: float
    net_quantity: float
    gross_turnover: float
    deal_vwap: float | None = None


class BulkDealsOut(BaseModel):
    """Disclosed bulk deals for one ticker (FR-15)."""

    ticker: str
    coverage: dict[str, Any]
    sources: list[str]
    deals: list[BulkDealOut]
    daily: list[BulkDealDailyOut]
    notes: list[str]


class BulkDealSymbolOut(BaseModel):
    symbol: str
    ticker: str
    security_name: str
    deal_rows: int
    sessions_with_deals: int
    first_deal: date
    last_deal: date
    gross_turnover: float
    sufficient: bool


# -- watchlist and history -------------------------------------------------
class WatchlistCreate(BaseModel):
    ticker: str = Field(..., max_length=20)
    note: str | None = Field(None, max_length=255)

    @field_validator("ticker")
    @classmethod
    def check_ticker(cls, value: str) -> str:
        return _validate_ticker(value)


class WatchlistItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ticker: str
    note: str | None
    created_at: datetime


class WatchlistQuoteOut(BaseModel):
    id: int
    ticker: str
    note: str | None
    company_name: str | None = None
    last_close: float | None = None
    change_percent: float | None = None
    last_date: date | None = None
    error: str | None = None


class PredictionRunOut(BaseModel):
    id: int
    ticker: str
    model_name: str
    forecast_horizon: int
    reference_date: date | None
    reference_price: float | None
    status: str
    auto_selected: bool
    data_source: str | None
    created_at: datetime
    runtime_seconds: float | None = None
    forecasts: list[ForecastStepOut] = Field(default_factory=list)
    metrics: MetricsOut | None = None
    error_message: str | None = None


# -- auth ------------------------------------------------------------------
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    created_at: datetime


class TokenOut(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    user: UserOut
