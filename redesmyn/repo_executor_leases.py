from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from redesmyn.db import RepoExecutorLease
from redesmyn.repo_identity import RepoKey

DEFAULT_LEASE_TTL = timedelta(seconds=60)


def _now() -> datetime:
    return datetime.now(UTC)


def _as_utc(dt: datetime) -> datetime:
    """
    Normalize datetimes for lease comparisons.

    SQLite frequently returns tz-naive datetimes even when SQLAlchemy models use
    `DateTime(timezone=True)`. We treat tz-naive values as UTC.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


async def get_primary_host_key(
    session: AsyncSession,
    repo: RepoKey,
    *,
    now: datetime | None = None,
) -> str | None:
    row = await session.get(RepoExecutorLease, (repo.workspace_id, repo.repo_id))
    if row is None:
        return None
    current = _as_utc(now or _now())
    if _as_utc(row.lease_expires_at) <= current:
        return None
    return row.host_key


async def acquire_or_refresh_primary(
    session: AsyncSession,
    repo: RepoKey,
    *,
    host_key: str,
    now: datetime | None = None,
    ttl: timedelta = DEFAULT_LEASE_TTL,
) -> bool:
    """
    Acquire the primary executor lease if free/expired, or refresh if already owned.

    Returns True if this host holds the lease after the call.
    """
    current = _as_utc(now or _now())
    expires_at = current + ttl
    row = await session.get(RepoExecutorLease, (repo.workspace_id, repo.repo_id))
    if row is None:
        session.add(
            RepoExecutorLease(
                workspace_id=repo.workspace_id,
                repo_id=repo.repo_id,
                host_key=host_key,
                lease_expires_at=expires_at,
            )
        )
        return True

    if _as_utc(row.lease_expires_at) <= current or row.host_key == host_key:
        row.host_key = host_key
        row.lease_expires_at = expires_at
        return True

    return False


async def expire_primary_if_owner(
    session: AsyncSession,
    repo: RepoKey,
    *,
    host_key: str,
    now: datetime | None = None,
) -> None:
    current = _as_utc(now or _now())
    row = await session.get(RepoExecutorLease, (repo.workspace_id, repo.repo_id))
    if row is None or row.host_key != host_key:
        return
    row.lease_expires_at = current
