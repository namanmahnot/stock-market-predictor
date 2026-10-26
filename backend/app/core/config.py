"""Application configuration, loaded from environment variables / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / "backend" / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Stock Market Price Prediction System"
    api_v1_prefix: str = "/api/v1"
    environment: str = "development"

    # SQLite keeps the lab demo zero-setup; point DATABASE_URL at PostgreSQL for
    # the deployment described in the SRS, e.g.
    # postgresql+psycopg://user:pass@localhost:5432/stockpred
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'data' / 'app.db').as_posix()}"

    # Auth
    # Long enough for HS256; replaced by JWT_SECRET in any real deployment.
    jwt_secret: str = "development-only-secret-replace-before-deployment"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7

    # CORS: restrict to the frontend origins in deployment (SEC-09).
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Data provider
    data_provider: str = "yfinance"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"
    cache_ttl_seconds: int = 3600
    artifact_dir: Path = PROJECT_ROOT / "data" / "artifacts"
    offline_csv_dir: Path = PROJECT_ROOT / "data" / "offline"
    alpha_vantage_api_key: str | None = None

    # NSE bulk-deal disclosures used as an optional feature group.
    bulk_deals_dir: Path = PROJECT_ROOT / "data" / "bulk_deals"
    #: Sessions with a disclosed deal before the feature group is worth using.
    bulk_deals_min_sessions: int = 10
    #: Usable rows required once the dataset is clipped to the covered window.
    bulk_deals_min_rows: int = 150
    #: Sessions to wait before a disclosure is treated as known. NSE publishes
    #: after the close, so 0 matches the real availability of the data.
    bulk_deals_publication_lag: int = 0

    # Modelling defaults
    default_history_days: int = 1095
    max_history_days: int = 3650
    min_observations: int = 250
    max_horizon: int = 30
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    walk_forward_step: int = 5
    max_walk_forward_origins: int = 40
    random_seed: int = 42

    # Rate limiting for expensive endpoints (SEC-07)
    heavy_endpoint_limit_per_minute: int = 6

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    return settings


settings = get_settings()
