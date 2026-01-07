"""Add milestone_id to linear_epic_defaults table.

Revision ID: 0021_linear_epic_defaults_milestone_id
Revises: 0020_rename_harness_profile_to_launch_configuration
Create Date: 2026-01-06
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0021_linear_epic_defaults_milestone_id"
down_revision = "0020_rename_harness_profile_to_launch_configuration"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "linear_epic_defaults" not in tables:
        return

    columns = {col["name"] for col in inspector.get_columns("linear_epic_defaults")}
    if "milestone_id" in columns:
        return

    op.add_column(
        "linear_epic_defaults",
        sa.Column("milestone_id", sa.String(), nullable=True),
    )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "linear_epic_defaults" not in tables:
        return

    columns = {col["name"] for col in inspector.get_columns("linear_epic_defaults")}
    if "milestone_id" not in columns:
        return

    op.drop_column("linear_epic_defaults", "milestone_id")
