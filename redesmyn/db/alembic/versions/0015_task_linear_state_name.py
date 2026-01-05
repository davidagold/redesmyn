"""Cache last observed Linear workflow state name on tasks.

Revision ID: 0015_task_linear_state_name
Revises: 0014_task_linear_state_cache
Create Date: 2026-01-05
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0015_task_linear_state_name"
down_revision = "0014_task_linear_state_cache"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "tasks" not in tables:
        return

    cols = {c["name"] for c in inspector.get_columns("tasks")}
    with op.batch_alter_table("tasks") as batch:
        if "linear_state_name" not in cols:
            batch.add_column(sa.Column("linear_state_name", sa.String(), nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "tasks" not in tables:
        return

    cols = {c["name"] for c in inspector.get_columns("tasks")}
    with op.batch_alter_table("tasks") as batch:
        if "linear_state_name" in cols:
            batch.drop_column("linear_state_name")
