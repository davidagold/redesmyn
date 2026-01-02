"""Git projections + merge run tables.

Revision ID: 0005_git_projections_tables
Revises: 0004_daemon_command_ack_data
Create Date: 2026-01-01
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0005_git_projections_tables"
down_revision = "0004_daemon_command_ack_data"
branch_labels = None
depends_on = None


def _json_type() -> sa.JSON:
    return sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "git_ref_states" not in tables:
        op.create_table(
            "git_ref_states",
            sa.Column("repository_id", sa.Integer(), nullable=False),
            sa.Column("refs", _json_type(), nullable=False),
            sa.Column(
                "observed_at",
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
            sa.ForeignKeyConstraint(["repository_id"], ["repositories.id"]),
            sa.PrimaryKeyConstraint("repository_id"),
        )

    if "git_trunk_timelines" not in tables:
        op.create_table(
            "git_trunk_timelines",
            sa.Column("epic_id", sa.Integer(), nullable=False),
            sa.Column("data", _json_type(), nullable=False),
            sa.Column(
                "observed_at",
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

    if "git_merge_bases" not in tables:
        op.create_table(
            "git_merge_bases",
            sa.Column("task_id", sa.Integer(), nullable=False),
            sa.Column("merge_base_sha", sa.String(), nullable=True),
            sa.Column(
                "observed_at",
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
            sa.ForeignKeyConstraint(["task_id"], ["tasks.id"]),
            sa.PrimaryKeyConstraint("task_id"),
        )

    if "merge_runs" not in tables:
        op.create_table(
            "merge_runs",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("run_id", sa.String(), nullable=False),
            sa.Column("epic_id", sa.Integer(), nullable=False),
            sa.Column("requested_task_id", sa.Integer(), nullable=False),
            sa.Column(
                "status",
                sa.Enum(
                    "running",
                    "blocked",
                    "resumable",
                    "succeeded",
                    "failed",
                    "canceled",
                    name="merge_run_status",
                    native_enum=False,
                ),
                nullable=False,
            ),
            sa.Column("scope", sa.String(), nullable=False),
            sa.Column("allow_running", sa.Boolean(), nullable=False),
            sa.Column("force", sa.Boolean(), nullable=False),
            sa.Column("plan", _json_type(), nullable=False),
            sa.Column("current_step_index", sa.Integer(), nullable=True),
            sa.Column("blocked_step_index", sa.Integer(), nullable=True),
            sa.Column("blocked_step_kind", sa.String(), nullable=True),
            sa.Column("blocked_task_id", sa.Integer(), nullable=True),
            sa.Column("blocked_branch_name", sa.String(), nullable=True),
            sa.Column("blocked_worktree_path", sa.String(), nullable=True),
            sa.Column("blocked_error", sa.Text(), nullable=True),
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
            sa.ForeignKeyConstraint(["requested_task_id"], ["tasks.id"]),
            sa.ForeignKeyConstraint(["blocked_task_id"], ["tasks.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            op.f("ix_merge_runs_run_id"), "merge_runs", ["run_id"], unique=True
        )
        op.create_index(op.f("ix_merge_runs_epic_id"), "merge_runs", ["epic_id"])
        op.create_index(
            op.f("ix_merge_runs_requested_task_id"),
            "merge_runs",
            ["requested_task_id"],
        )
        op.create_index(op.f("ix_merge_runs_status"), "merge_runs", ["status"])
        op.create_index(
            op.f("ix_merge_runs_blocked_task_id"), "merge_runs", ["blocked_task_id"]
        )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "merge_runs" in tables:
        op.drop_table("merge_runs")

    if "git_merge_bases" in tables:
        op.drop_table("git_merge_bases")

    if "git_trunk_timelines" in tables:
        op.drop_table("git_trunk_timelines")

    if "git_ref_states" in tables:
        op.drop_table("git_ref_states")
