"""Create tables for the SQLite demo database.

PostgreSQL deployments use the Alembic migration in backend/alembic instead.
"""
from __future__ import annotations

from app.db.models import Base
from app.db.session import engine


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
    print("Database schema created.")
