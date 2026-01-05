"""Cache last observed Linear issue state on tasks.

Revision ID: 0013_task_linear_state_cache
Revises: 0012_task_linear_identifier
Create Date: 2026-01-05
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0013_task_linear_state_cache"
down_revision = "0012_task_linear_identifier"
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
        if "linear_state_type" not in cols:
            batch.add_column(sa.Column("linear_state_type", sa.String(), nullable=True))
        if "linear_state_observed_at" not in cols:
            batch.add_column(
                sa.Column(
                    "linear_state_observed_at",
                    sa.DateTime(timezone=True),
                    nullable=True,
                )
            )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "tasks" not in tables:
        return

    cols = {c["name"] for c in inspector.get_columns("tasks")}
    with op.batch_alter_table("tasks") as batch:
        if "linear_state_observed_at" in cols:
            batch.drop_column("linear_state_observed_at")
        if "linear_state_type" in cols:
            batch.drop_column("linear_state_type")
