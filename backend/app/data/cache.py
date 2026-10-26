"""Time-to-live CSV cache for retrieved price history.

Caching keeps repeated dashboard requests inside the NFR-01 target and limits
the number of calls made to the external provider.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.core.config import settings


def _cache_path(key: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in key)
    return settings.cache_dir / f"{safe}.csv"


def read(key: str, ttl_seconds: int | None = None) -> tuple[pd.DataFrame, datetime] | None:
    """Return (frame, retrieved_at) when a fresh cache entry exists."""
    path = _cache_path(key)
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > (ttl_seconds if ttl_seconds is not None else settings.cache_ttl_seconds):
        return None
    try:
        frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    except (ValueError, KeyError, pd.errors.ParserError):
        return None
    retrieved_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return frame, retrieved_at


def write(key: str, frame: pd.DataFrame) -> None:
    path = _cache_path(key)
    out = frame.copy()
    out.index.name = "date"
    out.to_csv(path)


def stale(key: str) -> tuple[pd.DataFrame, datetime] | None:
    """Return an expired entry regardless of age, used when the provider fails."""
    return read(key, ttl_seconds=10**9)
