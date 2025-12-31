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

from redesmyn.db.migrate import stamp_revision, upgrade_to_head
from redesmyn.db.migrations.sqlite.agent_status_stopped import (
    migrate_agent_status_to_stopped,
)


def _sqlite_url(db_path: Path) -> str:
    return f"sqlite+aiosqlite:///{db_path}"


def create_engine(db_path: Path) -> AsyncEngine:
    return create_async_engine(_sqlite_url(db_path), future=True)


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def _sqlite_has_table(conn, *, name: str) -> bool:
    row = (
        await conn.exec_driver_sql(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,)
        )
    ).fetchone()
    return row is not None


async def init_db(engine: AsyncEngine) -> None:
    db_path_raw = engine.url.database
    if not db_path_raw:
        raise RuntimeError("Database URL is missing a path")

    db_path = Path(db_path_raw)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        upgrade_to_head(db_path=db_path)
        return

    should_stamp_baseline = False
    dialect_name: str | None = None
    has_alembic_version = False
    async with engine.begin() as conn:
        dialect_name = conn.dialect.name
        if dialect_name != "sqlite":
            return

        has_alembic_version = await _sqlite_has_table(conn, name="alembic_version")
        if has_alembic_version:
            return

        has_nodes = await _sqlite_has_table(conn, name="nodes")
        if has_nodes:
            # Pre-Alembic local DB: apply legacy SQLite migrations so the schema
            # matches the Alembic baseline before stamping.
            await _migrate_sqlite(conn)
            should_stamp_baseline = True

    if dialect_name != "sqlite":
        upgrade_to_head(db_path=db_path)
        return
    if has_alembic_version:
        upgrade_to_head(db_path=db_path)
        return
    if should_stamp_baseline:
        stamp_revision(db_path=db_path, revision="0001_baseline")
    upgrade_to_head(db_path=db_path)


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

    await migrate_agent_status_to_stopped(conn)

    result = await conn.exec_driver_sql("PRAGMA table_info(tasks)")
    existing = {row[1] for row in result.fetchall()}
    if "merge_ready_at" not in existing:
        await conn.execute(text("ALTER TABLE tasks ADD COLUMN merge_ready_at DATETIME"))


async def async_session(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with sessionmaker() as session:
        yield session
