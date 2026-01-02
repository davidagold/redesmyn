"""Add stack_in_sync column to tasks table.

Revision ID: 0008_add_task_stack_in_sync
Revises: 0007_drop_agent_sessions_node_id
Create Date: 2026-01-02
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0008_add_task_stack_in_sync"
down_revision = "0007_drop_agent_sessions_node_id"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "tasks" not in tables:
        return

    existing_cols = {c["name"] for c in inspector.get_columns("tasks")}
    if "stack_in_sync" in existing_cols:
        return

    with op.batch_alter_table("tasks") as batch:
        batch.add_column(sa.Column("stack_in_sync", sa.Boolean(), nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "tasks" not in tables:
        return

    existing_cols = {c["name"] for c in inspector.get_columns("tasks")}
    if "stack_in_sync" not in existing_cols:
        return

    with op.batch_alter_table("tasks") as batch:
        batch.drop_column("stack_in_sync")
