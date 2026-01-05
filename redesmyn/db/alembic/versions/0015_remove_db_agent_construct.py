"""Remove DB Agent construct.

Revision ID: 0015_remove_db_agent_construct
Revises: 0014_task_linear_state_cache
Create Date: 2026-01-05
"""

from __future__ import annotations

import json

from alembic import op
import sqlalchemy as sa

revision = "0015_remove_db_agent_construct"
down_revision = "0014_task_linear_state_cache"
branch_labels = None
depends_on = None


def _table_names(conn) -> set[str]:
    return set(sa.inspect(conn).get_table_names())


def _column_names(conn, table_name: str) -> set[str]:
    try:
        cols = sa.inspect(conn).get_columns(table_name)
    except Exception:
        return set()
    return {str(c.get("name")) for c in cols if c and c.get("name")}


def _task_id_by_agent_id(conn) -> dict[int, int]:
    if "tasks" not in _table_names(conn):
        return {}
    if "agent_id" not in _column_names(conn, "tasks"):
        return {}
    rows = conn.execute(
        sa.text("SELECT id, agent_id FROM tasks WHERE agent_id IS NOT NULL")
    ).mappings()
    mapping: dict[int, int] = {}
    for row in rows:
        task_id = row.get("id")
        agent_id = row.get("agent_id")
        if isinstance(task_id, int) and isinstance(agent_id, int):
            mapping[agent_id] = task_id
    return mapping


def _json_loads(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def upgrade() -> None:
    conn = op.get_bind()
    tables = _table_names(conn)

    task_id_by_agent_id = _task_id_by_agent_id(conn)

    # 1) Migrate Block.release acks payloads: required_agent_ids -> required_task_ids.
    if "blocks" in tables and "release" in _column_names(conn, "blocks"):
        rows = conn.execute(sa.text("SELECT id, release FROM blocks")).mappings()
        for row in rows:
            block_id = row.get("id")
            if not isinstance(block_id, int):
                continue
            release_raw = _json_loads(row.get("release"))
            if not isinstance(release_raw, dict):
                continue
            if release_raw.get("type") != "acks":
                continue
            if "required_task_ids" in release_raw:
                continue
            required_agent_ids = release_raw.pop("required_agent_ids", None)
            if not isinstance(required_agent_ids, list):
                continue
            required_task_ids: list[int] = []
            for agent_id in required_agent_ids:
                if not isinstance(agent_id, int):
                    continue
                task_id = task_id_by_agent_id.get(agent_id)
                if task_id is None:
                    continue
                required_task_ids.append(task_id)
            release_raw["required_task_ids"] = required_task_ids
            conn.execute(
                sa.text("UPDATE blocks SET release = :release WHERE id = :id"),
                {"id": block_id, "release": json.dumps(release_raw)},
            )

    # 2) Migrate block_acks: agent_id -> task_id.
    if "block_acks" in tables and "agent_id" in _column_names(conn, "block_acks"):
        op.create_table(
            "block_acks_new",
            sa.Column("block_id", sa.Integer(), nullable=False),
            sa.Column("task_id", sa.Integer(), nullable=False),
            sa.Column(
                "acked_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("(CURRENT_TIMESTAMP)"),
                nullable=False,
            ),
            sa.ForeignKeyConstraint(["block_id"], ["blocks.id"]),
            sa.ForeignKeyConstraint(["task_id"], ["tasks.id"]),
            sa.PrimaryKeyConstraint("block_id", "task_id"),
        )

        if task_id_by_agent_id:
            conn.execute(
                sa.text(
                    "INSERT INTO block_acks_new (block_id, task_id, acked_at) "
                    "SELECT ba.block_id, t.id, ba.acked_at "
                    "FROM block_acks AS ba "
                    "JOIN tasks AS t ON t.agent_id = ba.agent_id"
                )
            )
        op.drop_table("block_acks")
        op.rename_table("block_acks_new", "block_acks")

    # 3) Migrate agent_sessions: drop agent_id/agent_config_id; require task_id.
    if "agent_sessions" in tables:
        cols = _column_names(conn, "agent_sessions")
        if "agent_id" in cols:
            op.create_table(
                "agent_sessions_new",
                sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
                sa.Column("task_id", sa.Integer(), nullable=False),
                sa.Column("status", sa.String(), nullable=False),
                sa.Column("host_id", sa.Integer(), nullable=True),
                sa.Column("harness_profile_id", sa.String(), nullable=True),
                sa.Column("cwd_path", sa.String(), nullable=True),
                sa.Column("pid", sa.Integer(), nullable=True),
                sa.Column(
                    "attach",
                    sa.JSON(),
                    nullable=False,
                ),
                sa.Column("resolved_profile", sa.JSON(), nullable=True),
                sa.Column("exit_code", sa.Integer(), nullable=True),
                sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
                sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
                sa.Column("prelude_rendered", sa.Text(), nullable=True),
                sa.Column(
                    "created_at",
                    sa.DateTime(timezone=True),
                    server_default=sa.text("(CURRENT_TIMESTAMP)"),
                    nullable=False,
                ),
                sa.ForeignKeyConstraint(["task_id"], ["tasks.id"]),
                sa.ForeignKeyConstraint(["host_id"], ["hosts.id"]),
                sa.ForeignKeyConstraint(
                    ["harness_profile_id"], ["harness_profiles.id"]
                ),
                sa.PrimaryKeyConstraint("id"),
            )

            conn.execute(
                sa.text(
                    "INSERT INTO agent_sessions_new "
                    "(id, task_id, status, host_id, harness_profile_id, cwd_path, pid, attach, "
                    "resolved_profile, exit_code, started_at, ended_at, prelude_rendered, created_at) "
                    "SELECT "
                    "s.id, "
                    "COALESCE(s.task_id, (SELECT t.id FROM tasks AS t WHERE t.agent_id = s.agent_id LIMIT 1)) AS task_id, "
                    "s.status, s.host_id, s.harness_profile_id, s.cwd_path, s.pid, s.attach, "
                    "s.resolved_profile, s.exit_code, s.started_at, s.ended_at, s.prelude_rendered, s.created_at "
                    "FROM agent_sessions AS s "
                    "WHERE COALESCE(s.task_id, (SELECT t.id FROM tasks AS t WHERE t.agent_id = s.agent_id LIMIT 1)) IS NOT NULL"
                )
            )

            op.drop_table("agent_sessions")
            op.rename_table("agent_sessions_new", "agent_sessions")

            op.create_index("ix_agent_sessions_task_id", "agent_sessions", ["task_id"])
            op.create_index("ix_agent_sessions_host_id", "agent_sessions", ["host_id"])
            op.create_index(
                "ix_agent_sessions_harness_profile_id",
                "agent_sessions",
                ["harness_profile_id"],
            )
            conn.execute(
                sa.text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_sessions_active_task_id "
                    "ON agent_sessions (task_id) WHERE ended_at IS NULL"
                )
            )

    # 4) Drop tasks.agent_id.
    if "tasks" in tables and "agent_id" in _column_names(conn, "tasks"):
        task_indexes = {
            str(ix.get("name"))
            for ix in sa.inspect(conn).get_indexes("tasks")
            if ix and ix.get("name")
        }
        with op.batch_alter_table("tasks") as batch:
            if "ix_tasks_agent_id" in task_indexes:
                try:
                    batch.drop_index("ix_tasks_agent_id")
                except Exception:
                    pass
            try:
                batch.drop_constraint("fk_tasks_agent_id", type_="foreignkey")
            except Exception:
                pass
            batch.drop_column("agent_id")

    # 5) Drop commands.target_agent_id.
    if "commands" in tables and "target_agent_id" in _column_names(conn, "commands"):
        with op.batch_alter_table("commands") as batch:
            try:
                batch.drop_index("ix_commands_target_agent_id")
            except Exception:
                pass
            batch.drop_column("target_agent_id")

    # 6) Drop agent_configs + agents.
    if "agent_configs" in tables:
        op.drop_table("agent_configs")
    if "agents" in tables:
        op.drop_table("agents")


def downgrade() -> None:
    conn = op.get_bind()
    tables = _table_names(conn)

    if "tasks" not in tables:
        return

    # Recreate agents + task assignment (best-effort, per-task agents).
    if "agents" not in tables:
        op.create_table(
            "agents",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("display_name", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False, server_default="stopped"),
            sa.Column(
                "attach", sa.JSON(), nullable=False, server_default='{"type":"none"}'
            ),
            sa.Column("current_session_id", sa.Integer(), nullable=True),
            sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("(CURRENT_TIMESTAMP)"),
                nullable=False,
            ),
            sa.PrimaryKeyConstraint("id"),
        )

    if "agent_id" not in _column_names(conn, "tasks"):
        with op.batch_alter_table("tasks") as batch:
            batch.add_column(sa.Column("agent_id", sa.Integer(), nullable=True))
            batch.create_index("ix_tasks_agent_id", ["agent_id"])
            try:
                batch.create_foreign_key(
                    "fk_tasks_agent_id", "agents", ["agent_id"], ["id"]
                )
            except Exception:
                pass

        # Populate tasks.agent_id and create per-task agents.
        task_rows = conn.execute(sa.text("SELECT id FROM tasks ORDER BY id")).mappings()
        for row in task_rows:
            task_id = row.get("id")
            if not isinstance(task_id, int):
                continue
            result = conn.execute(
                sa.text("INSERT INTO agents (display_name) VALUES (:name)"),
                {"name": f"a-{task_id}"},
            )
            agent_id = int(result.lastrowid)  # type: ignore[attr-defined]
            conn.execute(
                sa.text("UPDATE tasks SET agent_id = :aid WHERE id = :tid"),
                {"aid": agent_id, "tid": task_id},
            )

    # Recreate agent_configs (empty).
    if "agent_configs" not in tables:
        op.create_table(
            "agent_configs",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("agent_id", sa.Integer(), nullable=False),
            sa.Column("harness_profile_id", sa.String(), nullable=True),
            sa.Column("definition", sa.JSON(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("(CURRENT_TIMESTAMP)"),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("(CURRENT_TIMESTAMP)"),
                nullable=False,
            ),
            sa.ForeignKeyConstraint(["agent_id"], ["agents.id"]),
            sa.ForeignKeyConstraint(["harness_profile_id"], ["harness_profiles.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("agent_id", name="uq_agent_configs_agent_id"),
        )
        op.create_index("ix_agent_configs_agent_id", "agent_configs", ["agent_id"])
        op.create_index(
            "ix_agent_configs_harness_profile_id",
            "agent_configs",
            ["harness_profile_id"],
        )

    # Restore commands.target_agent_id (null by default).
    if "commands" in tables and "target_agent_id" not in _column_names(
        conn, "commands"
    ):
        with op.batch_alter_table("commands") as batch:
            batch.add_column(sa.Column("target_agent_id", sa.Integer(), nullable=True))
            batch.create_index("ix_commands_target_agent_id", ["target_agent_id"])
            try:
                batch.create_foreign_key(None, "agents", ["target_agent_id"], ["id"])
            except Exception:
                pass

    # Restore block_acks.agent_id from task_id.
    if "block_acks" in tables and "task_id" in _column_names(conn, "block_acks"):
        op.create_table(
            "block_acks_old",
            sa.Column("block_id", sa.Integer(), nullable=False),
            sa.Column("agent_id", sa.Integer(), nullable=False),
            sa.Column(
                "acked_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("(CURRENT_TIMESTAMP)"),
                nullable=False,
            ),
            sa.ForeignKeyConstraint(["block_id"], ["blocks.id"]),
            sa.ForeignKeyConstraint(["agent_id"], ["agents.id"]),
            sa.PrimaryKeyConstraint("block_id", "agent_id"),
        )
        conn.execute(
            sa.text(
                "INSERT INTO block_acks_old (block_id, agent_id, acked_at) "
                "SELECT ba.block_id, t.agent_id, ba.acked_at "
                "FROM block_acks AS ba "
                "JOIN tasks AS t ON t.id = ba.task_id "
                "WHERE t.agent_id IS NOT NULL"
            )
        )
        op.drop_table("block_acks")
        op.rename_table("block_acks_old", "block_acks")

    # Restore blocks.release required_agent_ids.
    if "blocks" in tables and "release" in _column_names(conn, "blocks"):
        rows = conn.execute(sa.text("SELECT id, release FROM blocks")).mappings()
        for row in rows:
            block_id = row.get("id")
            if not isinstance(block_id, int):
                continue
            release_raw = _json_loads(row.get("release"))
            if not isinstance(release_raw, dict):
                continue
            if release_raw.get("type") != "acks":
                continue
            if "required_agent_ids" in release_raw:
                continue
            required_task_ids = release_raw.pop("required_task_ids", None)
            if not isinstance(required_task_ids, list):
                continue
            agent_ids: list[int] = []
            for task_id in required_task_ids:
                if not isinstance(task_id, int):
                    continue
                agent_id = conn.execute(
                    sa.text("SELECT agent_id FROM tasks WHERE id = :tid"),
                    {"tid": task_id},
                ).scalar()
                if isinstance(agent_id, int):
                    agent_ids.append(agent_id)
            release_raw["required_agent_ids"] = agent_ids
            conn.execute(
                sa.text("UPDATE blocks SET release = :release WHERE id = :id"),
                {"id": block_id, "release": json.dumps(release_raw)},
            )

    # Restore agent_sessions agent_id/agent_config_id and allow nullable task_id.
    if "agent_sessions" in tables and "task_id" in _column_names(
        conn, "agent_sessions"
    ):
        op.create_table(
            "agent_sessions_old",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("agent_id", sa.Integer(), nullable=False),
            sa.Column("agent_config_id", sa.Integer(), nullable=True),
            sa.Column("task_id", sa.Integer(), nullable=True),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("host_id", sa.Integer(), nullable=True),
            sa.Column("harness_profile_id", sa.String(), nullable=True),
            sa.Column("cwd_path", sa.String(), nullable=True),
            sa.Column("pid", sa.Integer(), nullable=True),
            sa.Column("attach", sa.JSON(), nullable=False),
            sa.Column("resolved_profile", sa.JSON(), nullable=True),
            sa.Column("exit_code", sa.Integer(), nullable=True),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("prelude_rendered", sa.Text(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("(CURRENT_TIMESTAMP)"),
                nullable=False,
            ),
            sa.ForeignKeyConstraint(["agent_id"], ["agents.id"]),
            sa.ForeignKeyConstraint(["agent_config_id"], ["agent_configs.id"]),
            sa.ForeignKeyConstraint(["task_id"], ["tasks.id"]),
            sa.ForeignKeyConstraint(["host_id"], ["hosts.id"]),
            sa.ForeignKeyConstraint(["harness_profile_id"], ["harness_profiles.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        conn.execute(
            sa.text(
                "INSERT INTO agent_sessions_old "
                "(id, agent_id, agent_config_id, task_id, status, host_id, harness_profile_id, cwd_path, pid, attach, "
                "resolved_profile, exit_code, started_at, ended_at, prelude_rendered, created_at) "
                "SELECT s.id, t.agent_id, NULL, s.task_id, s.status, s.host_id, s.harness_profile_id, s.cwd_path, s.pid, s.attach, "
                "s.resolved_profile, s.exit_code, s.started_at, s.ended_at, s.prelude_rendered, s.created_at "
                "FROM agent_sessions AS s "
                "JOIN tasks AS t ON t.id = s.task_id "
                "WHERE t.agent_id IS NOT NULL"
            )
        )
        op.drop_table("agent_sessions")
        op.rename_table("agent_sessions_old", "agent_sessions")

        op.create_index("ix_agent_sessions_agent_id", "agent_sessions", ["agent_id"])
        op.create_index("ix_agent_sessions_task_id", "agent_sessions", ["task_id"])
        op.create_index("ix_agent_sessions_host_id", "agent_sessions", ["host_id"])
        op.create_index(
            "ix_agent_sessions_harness_profile_id",
            "agent_sessions",
            ["harness_profile_id"],
        )
        conn.execute(
            sa.text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_sessions_active_agent_id "
                "ON agent_sessions (agent_id) WHERE ended_at IS NULL"
            )
        )
