"""Add label_name to linear_epic_defaults table.

Revision ID: 0022_linear_epic_defaults_label_name
Revises: 0021_linear_epic_defaults_milestone_id
Create Date: 2026-01-07
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0022_linear_epic_defaults_label_name"
down_revision = "0021_linear_epic_defaults_milestone_id"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "linear_epic_defaults" not in tables:
        return

    columns = {col["name"] for col in inspector.get_columns("linear_epic_defaults")}
    if "label_name" in columns:
        return

    op.add_column(
        "linear_epic_defaults",
        sa.Column("label_name", sa.String(), nullable=True),
    )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "linear_epic_defaults" not in tables:
        return

    columns = {col["name"] for col in inspector.get_columns("linear_epic_defaults")}
    if "label_name" not in columns:
        return

    op.drop_column("linear_epic_defaults", "label_name")
