from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from redesmyn.db.migrate import head_revision, upgrade_to_head


def _sqlite_url(db_path: Path) -> str:
    return f"sqlite+aiosqlite:///{db_path}"


SQLITE_BUSY_TIMEOUT_S = 30


def create_engine(db_path: Path) -> AsyncEngine:
    return create_async_engine(
        _sqlite_url(db_path),
        future=True,
        connect_args={"timeout": SQLITE_BUSY_TIMEOUT_S},
    )


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


class DatabaseNotInitializedError(RuntimeError):
    pass


class DatabaseMigrationRequiredError(RuntimeError):
    pass


async def _sqlite_alembic_version(conn) -> str | None:
    row = (
        await conn.exec_driver_sql("SELECT version_num FROM alembic_version")
    ).fetchone()
    if not row:
        return None
    value = row[0]
    return str(value) if value is not None else None


async def _sqlite_table_names(conn) -> set[str]:
    rows = (
        await conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'")
    ).fetchall()
    return {str(row[0]) for row in rows if row and row[0]}


async def init_db(engine: AsyncEngine, *, migrate: bool) -> None:
    db_path_raw = engine.url.database
    if not db_path_raw:
        raise RuntimeError("Database URL is missing a path")

    db_path = Path(db_path_raw)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        if not migrate:
            raise DatabaseNotInitializedError(
                f"Database not initialized at {db_path}. Run `rn init`."
            )
        upgrade_to_head(db_path=db_path)
        return

    dialect_name: str | None = None
    has_alembic_version = False
    has_any_tables = False
    alembic_version: str | None = None

    async with engine.begin() as conn:
        dialect_name = conn.dialect.name
        if dialect_name != "sqlite":
            # For now, Redesmyn's Alembic config is file-path based (SQLite).
            # If/when we add other dialects, we'll need a different config path.
            return

        # SQLite concurrency: use WAL for better reader/writer behavior.
        try:
            await conn.exec_driver_sql("PRAGMA journal_mode=WAL")
        except Exception:
            # Best-effort: older/unsupported SQLite builds may reject WAL.
            pass

        tables = await _sqlite_table_names(conn)
        has_alembic_version = "alembic_version" in tables
        has_any_tables = bool(tables)
        if not has_alembic_version and has_any_tables:
            # Pre-Alembic local DBs are no longer supported. Require manual intervention
            # so we don't silently drop or corrupt legacy data.
            raise DatabaseMigrationRequiredError(
                "Database is missing Alembic metadata (pre-Alembic DBs are unsupported). "
                f"Delete {db_path} and run `rn init` to recreate it."
            )

        if has_alembic_version:
            alembic_version = await _sqlite_alembic_version(conn)

    if not migrate:
        if not has_alembic_version:
            if not has_any_tables:
                raise DatabaseNotInitializedError(
                    f"Database not initialized at {db_path}. Run `rn init`."
                )
            raise DatabaseMigrationRequiredError(
                "Database is missing Alembic metadata. "
                f"Delete {db_path} and run `rn init` to recreate it."
            )
        try:
            head = head_revision(db_path=db_path)
        except RuntimeError as e:
            raise DatabaseMigrationRequiredError(str(e)) from e
        if alembic_version != head:
            raise DatabaseMigrationRequiredError(
                f"Database schema is out of date (current={alembic_version or 'unknown'}, head={head}). "
                "Run `rn daemon run` (or `just dev`) to migrate it."
            )
        return

    # Only upgrade when explicitly requested (typically on `rn daemon run` / `just dev`).
    upgrade_to_head(db_path=db_path)


async def async_session(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with sessionmaker() as session:
        yield session
