"""Add linear_epic_defaults table.

Revision ID: 0009_linear_epic_defaults
Revises: 0008_add_task_stack_in_sync
Create Date: 2026-01-03
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0009_linear_epic_defaults"
down_revision = "0008_add_task_stack_in_sync"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "linear_epic_defaults" in tables:
        return

    if "epics" not in tables:
        return

    op.create_table(
        "linear_epic_defaults",
        sa.Column("epic_id", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.String(), nullable=True),
        sa.Column("label_id", sa.String(), nullable=True),
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
        sa.ForeignKeyConstraint(["epic_id"], ["epics.id"]),
        sa.PrimaryKeyConstraint("epic_id"),
    )
    op.create_index(
        op.f("ix_linear_epic_defaults_epic_id"),
        "linear_epic_defaults",
        ["epic_id"],
    )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "linear_epic_defaults" not in tables:
        return

    op.drop_table("linear_epic_defaults")
