"""Require a branch before setting merge_ready_at.

Revision ID: 0013_tasks_merge_ready_requires_branch
Revises: 0012_task_linear_identifier
Create Date: 2026-01-05
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0013_tasks_merge_ready_requires_branch"
down_revision = "0012_task_linear_identifier"
branch_labels = None
depends_on = None

CONSTRAINT_NAME = "ck_tasks_merge_ready_requires_branch"
CONSTRAINT_SQL = "merge_ready_at IS NULL OR branch_name IS NOT NULL"


def _has_constraint(inspector: sa.Inspector, table_name: str, name: str) -> bool:
    try:
        constraints = inspector.get_check_constraints(table_name)
    except Exception:
        return False
    return any(constraint.get("name") == name for constraint in constraints)


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    if "tasks" not in set(inspector.get_table_names()):
        return

    conn.execute(
        sa.text(
            "UPDATE tasks SET merge_ready_at = NULL "
            "WHERE branch_name IS NULL AND merge_ready_at IS NOT NULL"
        )
    )

    if _has_constraint(inspector, "tasks", CONSTRAINT_NAME):
        return

    with op.batch_alter_table("tasks") as batch:
        batch.create_check_constraint(CONSTRAINT_NAME, CONSTRAINT_SQL)


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    if "tasks" not in set(inspector.get_table_names()):
        return

    if not _has_constraint(inspector, "tasks", CONSTRAINT_NAME):
        return

    with op.batch_alter_table("tasks") as batch:
        batch.drop_constraint(CONSTRAINT_NAME, type_="check")
