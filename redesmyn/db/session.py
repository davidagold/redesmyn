from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from redesmyn.db.models import Base


def _sqlite_url(db_path: Path) -> str:
    return f"sqlite+aiosqlite:///{db_path}"


def create_engine(db_path: Path) -> AsyncEngine:
    return create_async_engine(_sqlite_url(db_path), future=True)


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        if conn.dialect.name == "sqlite":
            await _migrate_sqlite(conn)


async def _migrate_sqlite(conn) -> None:
    result = await conn.exec_driver_sql("PRAGMA table_info(agents)")
    existing = {row[1] for row in result.fetchall()}

    attach_default = '{"type":"none"}'
    columns: list[tuple[str, str]] = [
        ("host_id", "INTEGER"),
        ("harness_profile_id", "VARCHAR"),
        ("cwd_path", "VARCHAR"),
        ("pid", "INTEGER"),
        ("attach", f"JSON NOT NULL DEFAULT '{attach_default}'"),
        ("resolved_profile", "JSON"),
        ("exit_code", "INTEGER"),
        ("started_at", "DATETIME"),
        ("ended_at", "DATETIME"),
    ]
    for name, ddl in columns:
        if name in existing:
            continue
        await conn.execute(text(f"ALTER TABLE agents ADD COLUMN {name} {ddl}"))


async def async_session(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with sessionmaker() as session:
        yield session
