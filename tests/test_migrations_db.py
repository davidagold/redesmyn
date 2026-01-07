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
def test_migrations_remove_db_agent_construct(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    _upgrade_to_revision(db_path, "0013_tasks_merge_ready_requires_branch")

    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        # Seed minimal rows needed to exercise the data migration.
        conn.execute(
            text(
                "INSERT INTO repositories (workspace_id, repo_id, repo_root, default_branch) "
                "VALUES (:workspace_id, :repo_id, :repo_root, :default_branch)"
            ),
            {
                "workspace_id": "default",
                "repo_id": "test",
                "repo_root": "/tmp/test",
                "default_branch": "main",
            },
        )
        repo_id = int(conn.execute(text("SELECT id FROM repositories")).scalar_one())
        conn.execute(
            text(
                "INSERT INTO epics (repository_id, name, slug, root_branch) "
                "VALUES (:repository_id, :name, :slug, :root_branch)"
            ),
            {
                "repository_id": repo_id,
                "name": "Test Epic",
                "slug": "test-epic",
                "root_branch": "main",
            },
        )
        epic_id = int(conn.execute(text("SELECT id FROM epics")).scalar_one())

        insert_agent = conn.execute(
            text(
                "INSERT INTO agents (display_name, status, attach) "
                "VALUES (:display_name, :status, :attach)"
            ),
            {
                "display_name": "Agent 1",
                "status": "stopped",
                "attach": '{"type":"none"}',
            },
        )
        agent_id = int(insert_agent.lastrowid)
        conn.execute(
            text(
                "INSERT INTO tasks (epic_id, title, agent_id, source, authority, state) "
                "VALUES (:epic_id, :title, :agent_id, :source, :authority, :state)"
            ),
            {
                "epic_id": epic_id,
                "title": "Task 1",
                "agent_id": agent_id,
                "source": "local",
                "authority": "local",
                "state": "todo",
            },
        )
        task_id = int(conn.execute(text("SELECT id FROM tasks")).scalar_one())

        # Create an agent session without task_id; 0014 should backfill it from tasks.agent_id.
        conn.execute(
            text(
                "INSERT INTO agent_sessions (agent_id, task_id, status, attach) "
                "VALUES (:agent_id, NULL, :status, :attach)"
            ),
            {"agent_id": agent_id, "status": "stopped", "attach": '{"type":"none"}'},
        )

    engine.dispose()

    upgrade_to_head(db_path=db_path)
    assert _sqlite_alembic_version(db_path) == head_revision(db_path=db_path)

    engine = create_engine(f"sqlite:///{db_path}")
    inspector = inspect(engine)

    assert "agents" not in set(inspector.get_table_names())
    assert "agent_configs" not in set(inspector.get_table_names())

    task_cols = {col["name"] for col in inspector.get_columns("tasks")}
    assert "agent_id" not in task_cols

    session_cols = {col["name"]: col for col in inspector.get_columns("agent_sessions")}
    assert "agent_id" not in session_cols
    assert "agent_config_id" not in session_cols
    assert session_cols["task_id"]["nullable"] is False
    assert session_cols["host_id"]["nullable"] is True
    assert session_cols["launch_configuration_id"]["nullable"] is True
    assert session_cols["resolved_launch_configuration"]["nullable"] is True
    assert session_cols["agent_kind_selection"]["nullable"] is False
    assert session_cols["agent_kind"]["nullable"] is False
    assert session_cols["agent_capabilities"]["nullable"] is False
    assert session_cols["agent_semantic_status"]["nullable"] is False
    assert session_cols["external_session_ref"]["nullable"] is False

    with engine.begin() as conn:
        # 0014 should have preserved the session row and filled task_id.
        row = conn.execute(
            text("SELECT task_id FROM agent_sessions ORDER BY id LIMIT 1")
        ).fetchone()
        assert row and int(row[0]) == task_id

        created_default = session_cols["created_at"].get("default")
        assert created_default and "CURRENT_TIMESTAMP" in str(created_default).upper()

        # New schema requires task_id, but host/profile fields remain nullable.
        conn.execute(
            text(
                "INSERT INTO agent_sessions (task_id, status, attach, ended_at) "
                "VALUES (:task_id, :status, :attach, CURRENT_TIMESTAMP)"
            ),
            {"task_id": task_id, "status": "stopped", "attach": '{"type":"none"}'},
        )


@pytest.mark.integration
def test_migrations_fix_agent_session_json_defaults(tmp_path: Path) -> None:
    db_path = tmp_path / "defaults.sqlite3"

    # 0017 had broken SQLite JSON defaults (colons interpreted as bind params).
    _upgrade_to_revision(db_path, "0017_agent_session_semantics")

    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO repositories (workspace_id, repo_id, repo_root, default_branch) "
                "VALUES (:workspace_id, :repo_id, :repo_root, :default_branch)"
            ),
            {
                "workspace_id": "default",
                "repo_id": "test",
                "repo_root": "/tmp/test",
                "default_branch": "main",
            },
        )
        repo_id = int(conn.execute(text("SELECT id FROM repositories")).scalar_one())
        conn.execute(
            text(
                "INSERT INTO epics (repository_id, name, slug, root_branch) "
                "VALUES (:repository_id, :name, :slug, :root_branch)"
            ),
            {
                "repository_id": repo_id,
                "name": "Test Epic",
                "slug": "test-epic",
                "root_branch": "main",
            },
        )
        epic_id = int(conn.execute(text("SELECT id FROM epics")).scalar_one())
        conn.execute(
            text(
                "INSERT INTO tasks (epic_id, title, source, authority, state) "
                "VALUES (:epic_id, :title, :source, :authority, :state)"
            ),
            {
                "epic_id": epic_id,
                "title": "Task 1",
                "source": "local",
                "authority": "local",
                "state": "todo",
            },
        )
        task_id = int(conn.execute(text("SELECT id FROM tasks")).scalar_one())

        # Insert a session relying on DB-side defaults for new JSON columns.
        conn.execute(
            text(
                "INSERT INTO agent_sessions (task_id, status, attach) "
                "VALUES (:task_id, :status, :attach)"
            ),
            {"task_id": task_id, "status": "running", "attach": '{"type":"none"}'},
        )

        row = conn.execute(
            text(
                "SELECT json_valid(agent_capabilities), json_valid(agent_semantic_status) "
                "FROM agent_sessions ORDER BY id DESC LIMIT 1"
            )
        ).fetchone()
        assert row and int(row[0]) == 0 and int(row[1]) == 0

    engine.dispose()

    upgrade_to_head(db_path=db_path)
    assert _sqlite_alembic_version(db_path) == head_revision(db_path=db_path)

    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        row = conn.execute(
            text(
                "SELECT json_valid(agent_capabilities), json_valid(agent_semantic_status) "
                "FROM agent_sessions ORDER BY id DESC LIMIT 1"
            )
        ).fetchone()
        assert row and int(row[0]) == 1 and int(row[1]) == 1

        create_sql = conn.execute(
            text(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='agent_sessions'"
            )
        ).scalar_one()
        assert '"can_send_text":true' in str(create_sql)
        assert '"detail":null' in str(create_sql)
