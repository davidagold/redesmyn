"""Create agent_sessions table (missing from baseline chain).

Revision ID: 0011_create_agent_sessions
Revises: 0010_linear_epic_defaults
Create Date: 2026-01-04
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0011_create_agent_sessions"
down_revision = "0010_linear_epic_defaults"
branch_labels = None
depends_on = None


def _json_type() -> sa.JSON:
    return sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def _dialect_server_default_json(conn, value: str) -> sa.TextClause:
    if conn.dialect.name == "postgresql":
        return sa.text(f"'{value}'::jsonb")
    return sa.text(f"'{value}'")


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "agent_sessions" in tables:
        return

    op.create_table(
        "agent_sessions",
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
            server_default=_dialect_server_default_json(conn, '{"type":"none"}'),
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
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "agent_sessions" not in tables:
        return

    op.drop_table("agent_sessions")
