"""Shared FastAPI dependencies."""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.security import decode_access_token
from app.db.models import User
from app.db.session import get_db

bearer_scheme = HTTPBearer(auto_error=False)


def _user_from_token(
    credentials: HTTPAuthorizationCredentials | None, db: Session
) -> User | None:
    if credentials is None or not credentials.credentials:
        return None
    subject = decode_access_token(credentials.credentials)
    if subject is None:
        return None
    user = db.get(User, int(subject)) if subject.isdigit() else None
    return user if user and user.is_active else None


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Require a valid token; used by every endpoint that touches user data."""
    user = _user_from_token(credentials, db)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid access token is required for this resource.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User | None:
    """Allow anonymous access, but attribute the request when a token is sent."""
    return _user_from_token(credentials, db)
