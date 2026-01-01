"""Daemon presence: host_key identity + tables.

Revision ID: 0003_daemon_presence_host_key
Revises: 0002_merge_node_into_task
Create Date: 2026-01-01
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

from sqlalchemy.dialects import postgresql

revision = "0003_daemon_presence_host_key"
down_revision = "0002_merge_node_into_task"
branch_labels = None
depends_on = None


def _json_type() -> sa.JSON:
    return sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "daemon_connections" not in tables:
        op.create_table(
            "daemon_connections",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("host_key", sa.String(), nullable=False),
            sa.Column("display_name", sa.String(), nullable=True),
            sa.Column("capabilities", _json_type(), nullable=False),
            sa.Column("attached_repos", _json_type(), nullable=False),
            sa.Column("connected_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("disconnected_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("disconnect_reason", sa.String(), nullable=True),
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
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            op.f("ix_daemon_connections_host_key"),
            "daemon_connections",
            ["host_key"],
            unique=False,
        )
    else:
        cols = {c["name"] for c in inspector.get_columns("daemon_connections")}
        with op.batch_alter_table("daemon_connections") as batch:
            if "daemon_id" in cols and "host_key" not in cols:
                batch.alter_column("daemon_id", new_column_name="host_key")
            if "host" in cols and "display_name" not in cols:
                batch.alter_column("host", new_column_name="display_name")

    if "daemon_commands" not in tables:
        op.create_table(
            "daemon_commands",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("host_key", sa.String(), nullable=False),
            sa.Column("command_type", sa.String(), nullable=False),
            sa.Column("workspace_id", sa.String(), nullable=True),
            sa.Column("repo_id", sa.String(), nullable=True),
            sa.Column("data", _json_type(), nullable=False),
            sa.Column(
                "state",
                sa.Enum(
                    "queued",
                    "running",
                    "succeeded",
                    "failed",
                    "canceled",
                    name="daemon_command_state",
                    native_enum=False,
                ),
                nullable=False,
            ),
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
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            op.f("ix_daemon_commands_host_key"),
            "daemon_commands",
            ["host_key"],
            unique=False,
        )
    else:
        cols = {c["name"] for c in inspector.get_columns("daemon_commands")}
        with op.batch_alter_table("daemon_commands") as batch:
            if "daemon_id" in cols and "host_key" not in cols:
                batch.alter_column("daemon_id", new_column_name="host_key")


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "daemon_commands" in tables:
        op.drop_table("daemon_commands")

    if "daemon_connections" in tables:
        op.drop_table("daemon_connections")
