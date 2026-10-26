"""In-memory fixed-window rate limiter for expensive endpoints (SRS 16.7).

A process-local limiter is enough for a lab prototype; a shared store such as
Redis would be required once the API runs in more than one worker.
"""
from __future__ import annotations

import threading
import time

from fastapi import HTTPException, Request, status

from app.core.config import settings

_WINDOW_SECONDS = 60
_hits: dict[str, list[float]] = {}
_lock = threading.Lock()


def _client_key(request: Request) -> str:
    client = request.client.host if request.client else "unknown"
    return f"{client}:{request.url.path}"


def enforce_rate_limit(request: Request) -> None:
    """FastAPI dependency: reject callers over the per-minute budget."""
    limit = settings.heavy_endpoint_limit_per_minute
    if limit <= 0:
        return
    key = _client_key(request)
    now = time.time()
    with _lock:
        recent = [stamp for stamp in _hits.get(key, []) if now - stamp < _WINDOW_SECONDS]
        if len(recent) >= limit:
            retry_after = int(_WINDOW_SECONDS - (now - recent[0])) + 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"This endpoint trains models and is limited to {limit} requests per minute. "
                    f"Try again in {retry_after}s."
                ),
                headers={"Retry-After": str(retry_after)},
            )
        recent.append(now)
        _hits[key] = recent


def reset_rate_limits() -> None:
    with _lock:
        _hits.clear()
