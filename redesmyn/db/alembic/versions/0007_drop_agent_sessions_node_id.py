"""Drop node_id column from agent_sessions.

The node_id column was a vestige from when nodes and tasks were separate
entities. Since the merge (0002), node_id has always mirrored task_id
and is redundant.

Revision ID: 0007_drop_agent_sessions_node_id
Revises: 0006_agent_sessions_v1
Create Date: 2026-01-02
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0007_drop_agent_sessions_node_id"
down_revision = "0006_agent_sessions_v1"
branch_labels = None
depends_on = None


def _json_type() -> sa.JSON:
    return sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "agent_sessions" not in tables:
        return

    existing_cols = {c["name"] for c in inspector.get_columns("agent_sessions")}
    if "node_id" not in existing_cols:
        return

    # SQLite: rebuild the table without node_id.
    op.create_table(
        "agent_sessions__v2",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("agent_id", sa.Integer(), nullable=False),
        sa.Column("agent_config_id", sa.Integer(), nullable=True),
        sa.Column("task_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("host_id", sa.Integer(), nullable=True),
        sa.Column("harness_profile_id", sa.String(), nullable=True),
        sa.Column("cwd_path", sa.String(), nullable=True),
        sa.Column("pid", sa.Integer(), nullable=True),
        sa.Column(
            "attach",
            _json_type(),
            nullable=False,
            server_default=sa.text('\'{"type":"none"}\''),
        ),
        sa.Column("resolved_profile", _json_type(), nullable=True),
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
        sa.Column(
            "updated_at",
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
            "INSERT INTO agent_sessions__v2 "
            "(id, agent_id, agent_config_id, task_id, status, host_id, harness_profile_id, "
            "cwd_path, pid, attach, resolved_profile, exit_code, started_at, ended_at, "
            "prelude_rendered, created_at, updated_at) "
            "SELECT "
            "id, agent_id, agent_config_id, task_id, status, host_id, harness_profile_id, "
            "cwd_path, pid, attach, resolved_profile, exit_code, started_at, ended_at, "
            "prelude_rendered, created_at, updated_at "
            "FROM agent_sessions"
        )
    )

    op.drop_table("agent_sessions")
    op.rename_table("agent_sessions__v2", "agent_sessions")

    # Recreate indexes (without node_id).
    op.create_index(op.f("ix_agent_sessions_agent_id"), "agent_sessions", ["agent_id"])
    op.create_index(
        op.f("ix_agent_sessions_agent_config_id"),
        "agent_sessions",
        ["agent_config_id"],
    )
    op.create_index(op.f("ix_agent_sessions_task_id"), "agent_sessions", ["task_id"])
    op.create_index(op.f("ix_agent_sessions_host_id"), "agent_sessions", ["host_id"])
    op.create_index(
        op.f("ix_agent_sessions_harness_profile_id"),
        "agent_sessions",
        ["harness_profile_id"],
    )

    # Only one active session per agent (ended_at is null).
    op.execute(
        sa.text(
            "CREATE UNIQUE INDEX uq_agent_sessions_active_agent "
            "ON agent_sessions (agent_id) WHERE ended_at IS NULL"
        )
    )
    # Only one active session per task (ended_at is null, task_id is not null).
    op.execute(
        sa.text(
            "CREATE UNIQUE INDEX uq_agent_sessions_active_task "
            "ON agent_sessions (task_id) WHERE ended_at IS NULL AND task_id IS NOT NULL"
        )
    )


def downgrade() -> None:
    # Re-add node_id column by rebuilding the table.
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "agent_sessions" not in tables:
        return

    op.create_table(
        "agent_sessions__v1",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("agent_id", sa.Integer(), nullable=False),
        sa.Column("agent_config_id", sa.Integer(), nullable=True),
        sa.Column("task_id", sa.Integer(), nullable=True),
        sa.Column("node_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("host_id", sa.Integer(), nullable=True),
        sa.Column("harness_profile_id", sa.String(), nullable=True),
        sa.Column("cwd_path", sa.String(), nullable=True),
        sa.Column("pid", sa.Integer(), nullable=True),
        sa.Column(
            "attach",
            _json_type(),
            nullable=False,
            server_default=sa.text('\'{"type":"none"}\''),
        ),
        sa.Column("resolved_profile", _json_type(), nullable=True),
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
        sa.Column(
            "updated_at",
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
            "INSERT INTO agent_sessions__v1 "
            "(id, agent_id, agent_config_id, task_id, node_id, status, host_id, harness_profile_id, "
            "cwd_path, pid, attach, resolved_profile, exit_code, started_at, ended_at, "
            "prelude_rendered, created_at, updated_at) "
            "SELECT "
            "id, agent_id, agent_config_id, task_id, task_id, status, host_id, harness_profile_id, "
            "cwd_path, pid, attach, resolved_profile, exit_code, started_at, ended_at, "
            "prelude_rendered, created_at, updated_at "
            "FROM agent_sessions"
        )
    )

    op.drop_table("agent_sessions")
    op.rename_table("agent_sessions__v1", "agent_sessions")

    # Recreate indexes (with node_id).
    op.create_index(op.f("ix_agent_sessions_agent_id"), "agent_sessions", ["agent_id"])
    op.create_index(
        op.f("ix_agent_sessions_agent_config_id"),
        "agent_sessions",
        ["agent_config_id"],
    )
    op.create_index(op.f("ix_agent_sessions_task_id"), "agent_sessions", ["task_id"])
    op.create_index(op.f("ix_agent_sessions_node_id"), "agent_sessions", ["node_id"])
    op.create_index(op.f("ix_agent_sessions_host_id"), "agent_sessions", ["host_id"])
    op.create_index(
        op.f("ix_agent_sessions_harness_profile_id"),
        "agent_sessions",
        ["harness_profile_id"],
    )

    op.execute(
        sa.text(
            "CREATE UNIQUE INDEX uq_agent_sessions_active_agent "
            "ON agent_sessions (agent_id) WHERE ended_at IS NULL"
        )
    )
    op.execute(
        sa.text(
            "CREATE UNIQUE INDEX uq_agent_sessions_active_task "
            "ON agent_sessions (task_id) WHERE ended_at IS NULL AND task_id IS NOT NULL"
        )
    )
