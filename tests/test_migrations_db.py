from __future__ import annotations

from pathlib import Path
import sqlite3

from alembic import command
import pytest
from sqlalchemy import create_engine, inspect, text

from redesmyn.db.migrate import alembic_config_for_db, head_revision, upgrade_to_head
from redesmyn.db.models import Base


def _upgrade_to_revision(db_path: Path, revision: str) -> None:
    command.upgrade(alembic_config_for_db(db_path), revision)


def _sqlite_alembic_version(db_path: Path) -> str:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT version_num FROM alembic_version").fetchone()
    assert row and row[0]
    return str(row[0])


@pytest.mark.integration
def test_migrations_apply_cleanly_and_match_orm_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "redesmyn.sqlite3"
    upgrade_to_head(db_path=db_path)

    assert _sqlite_alembic_version(db_path) == head_revision(db_path=db_path)

    engine = create_engine(f"sqlite:///{db_path}")
    inspector = inspect(engine)

    actual_tables = set(inspector.get_table_names())
    expected_tables = set(Base.metadata.tables.keys())
    assert expected_tables <= actual_tables


@pytest.mark.integration
def test_migrations_relax_legacy_agent_sessions_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    _upgrade_to_revision(db_path, "0005_git_projections_tables")

    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        # Seed rows for FK targets used by the rebuilt schema.
        conn.execute(
            text(
                "INSERT INTO hosts (host_key, display_name, capabilities) "
                "VALUES (:host_key, :display_name, :capabilities)"
            ),
            {
                "host_key": "test-host-key",
                "display_name": "Test Host",
                "capabilities": "{}",
            },
        )
        conn.execute(
            text(
                "INSERT INTO harness_profiles "
                "(id, kind, source, display_name, definition) "
                "VALUES (:id, :kind, :source, :display_name, :definition)"
            ),
            {
                "id": "hp1",
                "kind": "local",
                "source": "builtin",
                "display_name": "Test Harness",
                "definition": '{"argv":["true"]}',
            },
        )
        conn.execute(
            text(
                "INSERT INTO agents "
                "(display_name, status, host_id, harness_profile_id, attach) "
                "VALUES (:display_name, :status, :host_id, :harness_profile_id, :attach)"
            ),
            {
                "display_name": "Agent 1",
                "status": "stopped",
                "host_id": 1,
                "harness_profile_id": "hp1",
                "attach": '{"type":"none"}',
            },
        )

        # Simulate a legacy agent_sessions table with overly strict NOT NULL columns.
        conn.exec_driver_sql(
            """
            CREATE TABLE agent_sessions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              agent_id INTEGER NOT NULL,
              node_id INTEGER NULL,
              status TEXT NOT NULL,
              host_id INTEGER NOT NULL,
              harness_profile_id TEXT NOT NULL,
              cwd_path TEXT NULL,
              pid INTEGER NULL,
              attach TEXT NOT NULL,
              resolved_profile TEXT NOT NULL,
              exit_code INTEGER NULL,
              started_at DATETIME NULL,
              ended_at DATETIME NULL,
              created_at DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP)
            )
            """
        )
        conn.execute(
            text(
                "INSERT INTO agent_sessions "
                "(agent_id, node_id, status, host_id, harness_profile_id, attach, resolved_profile, ended_at) "
                "VALUES (:agent_id, :node_id, :status, :host_id, :harness_profile_id, :attach, :resolved_profile, :ended_at)"
            ),
            {
                "agent_id": 1,
                "node_id": None,
                "status": "stopped",
                "host_id": 1,
                "harness_profile_id": "hp1",
                "attach": '{"type":"none"}',
                "resolved_profile": "{}",
                "ended_at": "2026-01-01 00:00:00",
            },
        )

    engine.dispose()

    upgrade_to_head(db_path=db_path)
    assert _sqlite_alembic_version(db_path) == head_revision(db_path=db_path)

    engine = create_engine(f"sqlite:///{db_path}")
    inspector = inspect(engine)

    cols = {col["name"]: col for col in inspector.get_columns("agent_sessions")}
    assert cols["host_id"]["nullable"] is True
    assert cols["harness_profile_id"]["nullable"] is True
    assert cols["resolved_profile"]["nullable"] is True

    created_default = cols["created_at"].get("default")
    assert created_default and "CURRENT_TIMESTAMP" in str(created_default).upper()

    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO agent_sessions (agent_id, status, attach) "
                "VALUES (:agent_id, :status, :attach)"
            ),
            {"agent_id": 1, "status": "stopped", "attach": '{"type":"none"}'},
        )

        count = conn.execute(text("SELECT COUNT(1) FROM agent_sessions")).scalar_one()
        assert int(count) == 2
