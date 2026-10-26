"""Shared fixtures.

The suite is deterministic and offline: price history comes from a synthetic
CSV written into a temporary directory and served through the CSV provider, so
no test depends on an external market data feed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

SYNTHETIC_TICKER = "TEST"
N_SESSIONS = 760
#: Sessions at the end of the synthetic history that the bulk-deal export covers.
BULK_WINDOW_SESSIONS = 220


def make_price_frame(n: int = N_SESSIONS, seed: int = 7) -> pd.DataFrame:
    """Geometric random walk with mild momentum, in OHLCV shape.

    A deterministic series keeps model tests reproducible while still being
    realistic enough that indicators and splits behave as they would on real
    market data.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n)

    shocks = rng.normal(0.0004, 0.013, size=n)
    momentum = np.zeros(n)
    for i in range(1, n):
        momentum[i] = 0.15 * momentum[i - 1] + shocks[i]
    close = 100.0 * np.exp(np.cumsum(momentum))

    open_ = close * (1.0 + rng.normal(0.0, 0.003, size=n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.004, size=n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.004, size=n)))
    volume = rng.integers(1_000_000, 9_000_000, size=n).astype("float64")

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


def indian_grouping(value: int) -> str:
    """Format an integer the way NSE exports do, e.g. 3,90,000."""
    text = str(int(value))
    if len(text) <= 3:
        return text
    head, tail = text[:-3], text[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups + [tail])


def make_bulk_deals_csv(
    path: Path, dates: list[pd.Timestamp], symbol: str = SYNTHETIC_TICKER
) -> Path:
    """Write an NSE-shaped bulk-deals export, BOM and padded headers included."""
    header = (
        '"Date ","Symbol ","Security Name ","Client Name ","Buy / Sell ",'
        '"Quantity Traded ","Trade Price / Wght. Avg. Price ","Remarks "\n'
    )
    rows = [header]
    for index, day in enumerate(dates):
        stamp = day.strftime("%d-%b-%Y").upper()
        quantity = 50_000 + index * 1_000
        price = 100.0 + index
        side = "BUY" if index % 2 == 0 else "SELL"
        rows.append(
            f'"{stamp}","{symbol}","Synthetic Test Limited","CLIENT {index % 3}",'
            f'"{side}","{indian_grouping(quantity)}","{price:.2f}","-"\n'
        )
    path.write_text("".join(rows), encoding="utf-8-sig")
    return path


@pytest.fixture(scope="session")
def price_frame() -> pd.DataFrame:
    return make_price_frame()


@pytest.fixture(scope="session")
def bulk_deal_dates(price_frame: pd.DataFrame) -> list[pd.Timestamp]:
    """Every third session of the covered window carries a disclosed deal."""
    window = price_frame.index[-BULK_WINDOW_SESSIONS:]
    return list(window[::3])


@pytest.fixture(scope="session")
def feature_frame(price_frame: pd.DataFrame) -> pd.DataFrame:
    from app.ml.features import build_feature_frame

    return build_feature_frame(price_frame)


@pytest.fixture(autouse=True)
def offline_provider(tmp_path_factory, price_frame: pd.DataFrame, monkeypatch) -> Path:
    """Point the application at a synthetic CSV instead of a live provider."""
    from app.core.config import settings

    directory = tmp_path_factory.mktemp("offline")
    frame = price_frame.copy()
    frame.index.name = "date"
    frame.to_csv(directory / f"{SYNTHETIC_TICKER}.csv")

    cache_dir = tmp_path_factory.mktemp("cache")
    monkeypatch.setattr(settings, "data_provider", "csv")
    monkeypatch.setattr(settings, "offline_csv_dir", directory)
    monkeypatch.setattr(settings, "cache_dir", cache_dir)
    monkeypatch.setattr(settings, "artifact_dir", tmp_path_factory.mktemp("artifacts"))
    return directory


@pytest.fixture(autouse=True)
def offline_bulk_deals(tmp_path_factory, price_frame: pd.DataFrame, bulk_deal_dates, monkeypatch) -> Path:
    """Serve a synthetic bulk-deal export instead of the real NSE download."""
    from app.core.config import settings
    from app.data import bulk_deals

    directory = tmp_path_factory.mktemp("bulk_deals")
    make_bulk_deals_csv(directory / "synthetic-bulk-deals.csv", bulk_deal_dates)
    monkeypatch.setattr(settings, "bulk_deals_dir", directory)
    bulk_deals.clear_cache()
    yield directory
    bulk_deals.clear_cache()


@pytest.fixture
def history(offline_provider):
    from app.data.providers import load_history

    return load_history(SYNTHETIC_TICKER, days=3650)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient backed by a throwaway SQLite database."""
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.db.models import Base
    from app.db.session import get_db
    from app.main import app
    from app.services.forecast_service import clear_experiment_cache
    from app.services.rate_limit import reset_rate_limits

    engine = create_engine(
        f"sqlite:///{(tmp_path / 'test.db').as_posix()}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

    def override_get_db():
        session = TestingSession()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    clear_experiment_cache()
    reset_rate_limits()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    clear_experiment_cache()
    engine.dispose()


@pytest.fixture
def auth_client(client):
    """A client with a registered user's bearer token applied."""
    response = client.post(
        "/api/v1/auth/register", json={"email": "student@example.com", "password": "LabPassword123"}
    )
    assert response.status_code == 201, response.text
    token = response.json()["access_token"]
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client
