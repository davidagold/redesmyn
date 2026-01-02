"""Agent session schema (v1).

Revision ID: 0006_agent_sessions_v1
Revises: 0005_git_projections_tables
Create Date: 2026-01-02
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0006_agent_sessions_v1"
down_revision = "0005_git_projections_tables"
branch_labels = None
depends_on = None


def _json_type() -> sa.JSON:
    return sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def _dialect_server_default_json(conn, value: str) -> sa.TextClause:
    if conn.dialect.name == "postgresql":
        return sa.text(f"'{value}'::jsonb")
    return sa.text(f"'{value}'")


def _add_column_if_missing(
    inspector, *, table: str, name: str, column: sa.Column
) -> None:
    cols = {c["name"] for c in inspector.get_columns(table)}
    if name in cols:
        return
    with op.batch_alter_table(table) as batch:
        batch.add_column(column)


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "agents" in tables:
        _add_column_if_missing(
            inspector,
            table="agents",
            name="current_session_id",
            column=sa.Column("current_session_id", sa.Integer(), nullable=True),
        )
        cols = {c["name"] for c in inspector.get_columns("agents")}
        indexes = {idx["name"] for idx in inspector.get_indexes("agents")}
        if (
            "current_session_id" in cols
            and "ix_agents_current_session_id" not in indexes
        ):
            op.create_index(
                op.f("ix_agents_current_session_id"),
                "agents",
                ["current_session_id"],
                unique=False,
            )

    if "agent_configs" not in tables:
        op.create_table(
            "agent_configs",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("agent_id", sa.Integer(), nullable=False),
            sa.Column("harness_profile_id", sa.String(), nullable=True),
            sa.Column("definition", _json_type(), nullable=False),
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
        op.create_index(
            op.f("ix_agent_configs_agent_id"), "agent_configs", ["agent_id"]
        )
        op.create_index(
            op.f("ix_agent_configs_harness_profile_id"),
            "agent_configs",
            ["harness_profile_id"],
        )

    if "agent_sessions" not in tables:
        return

    existing_cols = {c["name"]: c for c in inspector.get_columns("agent_sessions")}
    existing_fks = inspector.get_foreign_keys("agent_sessions")
    has_nodes_fk = any(fk.get("referred_table") == "nodes" for fk in existing_fks)

    required_cols = {"agent_config_id", "task_id", "prelude_rendered"}
    missing_required = sorted(required_cols - set(existing_cols))

    should_rebuild = bool(missing_required or has_nodes_fk)
    # If the legacy schema has strict NOT NULL constraints for fields that should be
    # optional (host/harness/profile), we rebuild to relax them.
    for col_name in ("host_id", "harness_profile_id", "resolved_profile"):
        col = existing_cols.get(col_name)
        if col is not None and not col.get("nullable", True):
            should_rebuild = True

    if not should_rebuild:
        return

    # SQLite: rebuild the table to add/relax columns and drop broken FKs (nodes).
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

    select_task_id = "task_id" if "task_id" in existing_cols else "node_id"
    select_resolved_profile = (
        "resolved_profile" if "resolved_profile" in existing_cols else "NULL"
    )
    select_updated_at = "updated_at" if "updated_at" in existing_cols else "created_at"

    conn.execute(
        sa.text(
            "INSERT INTO agent_sessions__v1 "
            "(id, agent_id, agent_config_id, task_id, node_id, status, host_id, harness_profile_id, "
            "cwd_path, pid, attach, resolved_profile, exit_code, started_at, ended_at, prelude_rendered, "
            "created_at, updated_at) "
            "SELECT "
            "id, agent_id, NULL, "
            f"{select_task_id} AS task_id, "
            "node_id, status, host_id, harness_profile_id, cwd_path, pid, attach, "
            f"{select_resolved_profile} AS resolved_profile, "
            "exit_code, started_at, ended_at, NULL, created_at, "
            f"{select_updated_at} "
            "FROM agent_sessions"
        )
    )

    op.drop_table("agent_sessions")
    op.rename_table("agent_sessions__v1", "agent_sessions")

    # Recreate key indexes (use our new naming patterns; keep old unique semantics).
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

    # Only one active session per agent (ended_at is null).
    op.execute(
        sa.text(
            "CREATE UNIQUE INDEX uq_agent_sessions_active_agent "
            "ON agent_sessions (agent_id) WHERE ended_at IS NULL"
        )
    )
    # Only one active session per task (node == task in v1).
    op.execute(
        sa.text(
            "CREATE UNIQUE INDEX uq_agent_sessions_active_task "
            "ON agent_sessions (task_id) "
            "WHERE ended_at IS NULL AND task_id IS NOT NULL"
        )
    )


def downgrade() -> None:
    # Best-effort: keep schema as-is (data model evolved quickly in v0).
    pass
