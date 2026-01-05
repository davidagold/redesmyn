from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
import json

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from redesmyn.db.migrate import head_revision, stamp_revision, upgrade_to_head
from redesmyn.db.migrations.sqlite.agent_status_stopped import (
    migrate_agent_status_to_stopped,
)


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


async def _sqlite_has_table(conn, *, name: str) -> bool:
    row = (
        await conn.exec_driver_sql(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,)
        )
    ).fetchone()
    return row is not None


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

    should_stamp_baseline = False
    dialect_name: str | None = None
    has_alembic_version = False
    alembic_version: str | None = None

    async with engine.begin() as conn:
        dialect_name = conn.dialect.name
        if dialect_name != "sqlite":
            # For now, Redesmyn's Alembic config is file-path based (SQLite).
            # If/when we add other dialects, we'll need a different config path.
            return

        # SQLite concurrency: use WAL for better reader/writer behavior and apply
        # a reasonable sync mode for local-dev durability/performance tradeoffs.
        try:
            await conn.exec_driver_sql("PRAGMA journal_mode=WAL")
            await conn.exec_driver_sql("PRAGMA synchronous=NORMAL")
        except Exception:
            # Best-effort: older/unsupported SQLite builds may reject WAL.
            pass

        has_alembic_version = await _sqlite_has_table(conn, name="alembic_version")
        if not has_alembic_version and await _sqlite_has_table(conn, name="nodes"):
            if not migrate:
                raise DatabaseMigrationRequiredError(
                    "Database is from a pre-Alembic Redesmyn version. Run `rn daemon run` "
                    "(or `rn dev`) to migrate it."
                )
            # Pre-Alembic local DB: apply legacy SQLite migrations so the schema
            # matches the Alembic baseline before stamping.
            await _migrate_sqlite(conn)
            should_stamp_baseline = True

        if has_alembic_version:
            alembic_version = await _sqlite_alembic_version(conn)

    if should_stamp_baseline:
        stamp_revision(db_path=db_path, revision="0001_baseline")

    if not migrate:
        if not has_alembic_version:
            raise DatabaseMigrationRequiredError(
                "Database is missing Alembic metadata. Run `rn daemon run` (or `rn dev`) to migrate it."
            )
        try:
            head = head_revision(db_path=db_path)
        except RuntimeError as e:
            raise DatabaseMigrationRequiredError(str(e)) from e
        if alembic_version != head:
            raise DatabaseMigrationRequiredError(
                f"Database schema is out of date (current={alembic_version or 'unknown'}, head={head}). "
                "Run `rn daemon run` (or `rn dev`) to migrate it."
            )
        return

    # Only upgrade when explicitly requested (typically on `rn daemon run` / `rn dev`).
    upgrade_to_head(db_path=db_path)


async def _migrate_sqlite(conn) -> None:
    result = await conn.exec_driver_sql("PRAGMA table_info(agents)")
    existing = {row[1] for row in result.fetchall()}

    if "current_session_id" not in existing:
        await conn.execute(
            text("ALTER TABLE agents ADD COLUMN current_session_id INTEGER")
        )

    await migrate_agent_status_to_stopped(conn)
    await _backfill_agent_sessions_from_legacy_agents(conn)

    result = await conn.exec_driver_sql("PRAGMA table_info(tasks)")
    existing = {row[1] for row in result.fetchall()}
    if "merge_ready_at" not in existing:
        await conn.execute(text("ALTER TABLE tasks ADD COLUMN merge_ready_at DATETIME"))


async def _backfill_agent_sessions_from_legacy_agents(conn) -> None:
    # Best-effort: if upgrading a local DB that used the pre-T-8 `agents` table as
    # the "latest run", populate `agent_configs` + `agent_sessions` once so
    # restart/log UX keeps working without manual intervention.
    agent_sessions_count = (
        await conn.exec_driver_sql("SELECT COUNT(1) FROM agent_sessions")
    ).fetchone()
    if agent_sessions_count and agent_sessions_count[0]:
        return

    agent_cols = (await conn.exec_driver_sql("PRAGMA table_info(agents)")).fetchall()
    columns = {row[1] for row in agent_cols if row and row[1]}
    legacy_cols = {
        "status",
        "host_id",
        "harness_profile_id",
        "cwd_path",
        "pid",
        "attach",
        "resolved_profile",
        "exit_code",
        "started_at",
        "ended_at",
    }
    if not (columns & legacy_cols):
        return

    select_cols = ["id", "display_name", *(sorted(columns & legacy_cols))]
    quoted = ", ".join(f'"{c}"' for c in select_cols)
    rows = (await conn.exec_driver_sql(f"SELECT {quoted} FROM agents")).fetchall()
    if not rows:
        return

    col_index = {name: idx for idx, name in enumerate(select_cols)}

    def get(row, key: str):
        idx = col_index.get(key)
        if idx is None:
            return None
        return row[idx]

    for row in rows:
        agent_id = get(row, "id")
        if not isinstance(agent_id, int):
            continue

        resolved_profile_raw = get(row, "resolved_profile")
        started_at = get(row, "started_at")
        ended_at = get(row, "ended_at")

        has_run_evidence = (
            resolved_profile_raw is not None
            or started_at is not None
            or ended_at is not None
        )
        if not has_run_evidence:
            continue

        harness_profile_id = get(row, "harness_profile_id")
        host_id = get(row, "host_id")
        cwd_path = get(row, "cwd_path")
        pid = get(row, "pid")
        attach_raw = get(row, "attach") or '{"type":"none"}'
        exit_code = get(row, "exit_code")
        status = get(row, "status") or "stopped"
        if status == "idle":
            status = "stopped"

        try:
            resolved_profile = (
                json.loads(resolved_profile_raw)
                if isinstance(resolved_profile_raw, str)
                else resolved_profile_raw
            )
        except Exception:
            resolved_profile = None

        try:
            attach = (
                json.loads(attach_raw) if isinstance(attach_raw, str) else attach_raw
            )
        except Exception:
            attach = {"type": "none"}

        node_row = (
            await conn.exec_driver_sql(
                "SELECT id, primary_task_id FROM nodes WHERE agent_id = ? ORDER BY id LIMIT 1",
                (agent_id,),
            )
        ).fetchone()
        task_id = node_row[1] if node_row else None

        agent_config_id = None
        if resolved_profile is not None:
            existing_config = (
                await conn.exec_driver_sql(
                    "SELECT id FROM agent_configs WHERE agent_id = ? LIMIT 1",
                    (agent_id,),
                )
            ).fetchone()
            if existing_config:
                agent_config_id = existing_config[0]
            else:
                await conn.exec_driver_sql(
                    "INSERT INTO agent_configs (agent_id, harness_profile_id, definition) VALUES (?, ?, ?)",
                    (agent_id, harness_profile_id, json.dumps(resolved_profile)),
                )
                agent_config_id = (
                    await conn.exec_driver_sql("SELECT last_insert_rowid()")
                ).fetchone()[0]

        await conn.exec_driver_sql(
            "INSERT INTO agent_sessions "
            "(agent_id, agent_config_id, task_id, status, host_id, harness_profile_id, cwd_path, pid, attach, resolved_profile, exit_code, started_at, ended_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                agent_id,
                agent_config_id,
                task_id,
                status,
                host_id,
                harness_profile_id,
                cwd_path,
                pid,
                json.dumps(attach),
                json.dumps(resolved_profile) if resolved_profile is not None else None,
                exit_code,
                started_at,
                ended_at,
            ),
        )
        agent_session_id = (
            await conn.exec_driver_sql("SELECT last_insert_rowid()")
        ).fetchone()[0]

        is_active = status in {"running", "blocked"} and ended_at is None
        if is_active:
            await conn.exec_driver_sql(
                "UPDATE agents SET current_session_id = ? WHERE id = ?",
                (agent_session_id, agent_id),
            )


async def async_session(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with sessionmaker() as session:
        yield session
