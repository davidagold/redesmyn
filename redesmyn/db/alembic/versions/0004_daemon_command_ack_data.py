"""Persist daemon command ack data separately from payload.

Revision ID: 0004_daemon_command_ack_data
Revises: 0003_daemon_presence_host_key
Create Date: 2026-01-01
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

from sqlalchemy.dialects import postgresql

revision = "0004_daemon_command_ack_data"
down_revision = "0003_daemon_presence_host_key"
branch_labels = None
depends_on = None


def _json_type() -> sa.JSON:
    return sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())
    if "daemon_commands" not in tables:
        return

    cols = {c["name"] for c in inspector.get_columns("daemon_commands")}
    if "ack_data" in cols:
        return

    dialect = conn.dialect.name
    if dialect == "postgresql":
        server_default = sa.text("'{}'::jsonb")
    else:
        server_default = sa.text("'{}'")

    with op.batch_alter_table("daemon_commands") as batch:
        batch.add_column(
            sa.Column(
                "ack_data",
                _json_type(),
                nullable=False,
                server_default=server_default,
            )
        )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())
    if "daemon_commands" not in tables:
        return
    cols = {c["name"] for c in inspector.get_columns("daemon_commands")}
    if "ack_data" not in cols:
        return
    with op.batch_alter_table("daemon_commands") as batch:
        batch.drop_column("ack_data")
