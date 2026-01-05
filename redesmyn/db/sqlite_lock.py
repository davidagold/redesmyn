from __future__ import annotations

from sqlalchemy.exc import OperationalError


def is_sqlite_database_locked_error(exc: BaseException) -> bool:
    """
    Detect SQLite lock contention errors.

    SQLite uses a coarse database-level write lock. When a writer is active,
    other writers will fail with messages like "database is locked" once the
    connection busy timeout elapses.
    """

    if isinstance(exc, OperationalError):
        msg = str(getattr(exc, "orig", exc)).lower()
    else:
        msg = str(exc).lower()
    return "database is locked" in msg or "database schema is locked" in msg


def sqlite_lock_backoff_s(attempt: int) -> float:
    """
    Exponential backoff for retries after lock contention.

    Kept small because this is primarily for local-dev ergonomics.
    """

    attempt = max(0, attempt)
    base = 0.05
    cap = 1.0
    return min(cap, base * (2**min(attempt, 6)))

