"""GitHub repo association (epic defaults + task override).

Revision ID: 0024_github_repo_association
Revises: 0023_agent_interface_mode_and_preview
Create Date: 2026-01-10
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0024_github_repo_association"
down_revision = "0023_agent_interface_mode_and_preview"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("epics") as batch:
        batch.add_column(sa.Column("github_repo_host", sa.String(), nullable=True))
        batch.add_column(sa.Column("github_repo_owner", sa.String(), nullable=True))
        batch.add_column(sa.Column("github_repo_name", sa.String(), nullable=True))
        batch.create_check_constraint(
            "ck_epics_github_repo_complete",
            "(github_repo_host IS NULL AND github_repo_owner IS NULL AND github_repo_name IS NULL) "
            "OR (github_repo_host IS NOT NULL AND github_repo_owner IS NOT NULL AND github_repo_name IS NOT NULL)",
        )

    with op.batch_alter_table("tasks") as batch:
        batch.add_column(sa.Column("github_repo_host", sa.String(), nullable=True))
        batch.add_column(sa.Column("github_repo_owner", sa.String(), nullable=True))
        batch.add_column(sa.Column("github_repo_name", sa.String(), nullable=True))
        batch.create_check_constraint(
            "ck_tasks_github_repo_complete",
            "(github_repo_host IS NULL AND github_repo_owner IS NULL AND github_repo_name IS NULL) "
            "OR (github_repo_host IS NOT NULL AND github_repo_owner IS NOT NULL AND github_repo_name IS NOT NULL)",
        )


def downgrade() -> None:
    with op.batch_alter_table("tasks") as batch:
        batch.drop_constraint("ck_tasks_github_repo_complete", type_="check")
        batch.drop_column("github_repo_name")
        batch.drop_column("github_repo_owner")
        batch.drop_column("github_repo_host")

    with op.batch_alter_table("epics") as batch:
        batch.drop_constraint("ck_epics_github_repo_complete", type_="check")
        batch.drop_column("github_repo_name")
        batch.drop_column("github_repo_owner")
        batch.drop_column("github_repo_host")
