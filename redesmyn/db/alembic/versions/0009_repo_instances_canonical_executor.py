"""Repo instances + canonical executor routing (v1 schema).

Revision ID: 0009_repo_instances_canonical_executor
Revises: 0008_add_task_stack_in_sync
Create Date: 2026-01-03
"""

from __future__ import annotations

from hashlib import sha256

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0009_repo_instances_canonical_executor"
down_revision = "0008_add_task_stack_in_sync"
branch_labels = None
depends_on = None


def _json_type() -> sa.JSON:
    return sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "repositories" in tables:
        cols = {c["name"] for c in inspector.get_columns("repositories")}
        with op.batch_alter_table("repositories") as batch:
            if "workspace_id" not in cols:
                batch.add_column(
                    sa.Column(
                        "workspace_id",
                        sa.String(),
                        nullable=False,
                        server_default="default",
                    )
                )
            if "repo_id" not in cols:
                batch.add_column(sa.Column("repo_id", sa.String(), nullable=True))

        # Best-effort backfill for existing repos.
        inspector = sa.inspect(conn)
        rows = list(
            conn.execute(
                sa.text("SELECT id, repo_root, repo_id FROM repositories ORDER BY id")
            )
        )
        for row in rows:
            repo_id = row.repo_id
            if repo_id:
                continue
            repo_root = row.repo_root or ""
            computed = sha256(str(repo_root).encode("utf-8")).hexdigest()[:16]
            conn.execute(
                sa.text("UPDATE repositories SET repo_id = :repo_id WHERE id = :id"),
                {"repo_id": computed, "id": row.id},
            )

        inspector = sa.inspect(conn)
        cols = {c["name"] for c in inspector.get_columns("repositories")}
        if "repo_id" in cols:
            with op.batch_alter_table("repositories") as batch:
                batch.alter_column("repo_id", existing_type=sa.String(), nullable=False)

        try:
            op.create_unique_constraint(
                "uq_repositories_workspace_repo_id",
                "repositories",
                ["workspace_id", "repo_id"],
            )
        except Exception:
            # Constraint may already exist (SQLite batch mode / repeated migrations).
            pass

    if "repo_executor_leases" not in tables:
        op.create_table(
            "repo_executor_leases",
            sa.Column("workspace_id", sa.String(), nullable=False),
            sa.Column("repo_id", sa.String(), nullable=False),
            sa.Column("host_key", sa.String(), nullable=False),
            sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=False),
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
            sa.PrimaryKeyConstraint("workspace_id", "repo_id"),
        )
        op.create_index(
            op.f("ix_repo_executor_leases_host_key"),
            "repo_executor_leases",
            ["host_key"],
            unique=False,
        )

    if "git_ref_states_by_instance" not in tables:
        op.create_table(
            "git_ref_states_by_instance",
            sa.Column("repository_id", sa.Integer(), nullable=False),
            sa.Column("host_key", sa.String(), nullable=False),
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
            sa.PrimaryKeyConstraint("repository_id", "host_key"),
        )

    if "git_trunk_timelines_by_instance" not in tables:
        op.create_table(
            "git_trunk_timelines_by_instance",
            sa.Column("epic_id", sa.Integer(), nullable=False),
            sa.Column("host_key", sa.String(), nullable=False),
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
            sa.PrimaryKeyConstraint("epic_id", "host_key"),
        )

    if "git_merge_bases_by_instance" not in tables:
        op.create_table(
            "git_merge_bases_by_instance",
            sa.Column("task_id", sa.Integer(), nullable=False),
            sa.Column("host_key", sa.String(), nullable=False),
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
            sa.PrimaryKeyConstraint("task_id", "host_key"),
        )

    if "task_stack_in_sync_states" not in tables:
        op.create_table(
            "task_stack_in_sync_states",
            sa.Column("task_id", sa.Integer(), nullable=False),
            sa.Column("host_key", sa.String(), nullable=False),
            sa.Column("stack_in_sync", sa.Boolean(), nullable=True),
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
            sa.PrimaryKeyConstraint("task_id", "host_key"),
        )

    if "merge_runs" in tables:
        cols = {c["name"] for c in inspector.get_columns("merge_runs")}
        with op.batch_alter_table("merge_runs") as batch:
            if "host_key" not in cols:
                batch.add_column(sa.Column("host_key", sa.String(), nullable=True))
            if "canonical" not in cols:
                batch.add_column(
                    sa.Column(
                        "canonical",
                        sa.Boolean(),
                        nullable=False,
                        server_default=sa.text("1"),
                    )
                )
        # Best-effort: clear server_default if supported.
        try:
            with op.batch_alter_table("merge_runs") as batch:
                batch.alter_column("canonical", server_default=None)
        except Exception:
            pass


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "merge_runs" in tables:
        cols = {c["name"] for c in inspector.get_columns("merge_runs")}
        with op.batch_alter_table("merge_runs") as batch:
            if "canonical" in cols:
                batch.drop_column("canonical")
            if "host_key" in cols:
                batch.drop_column("host_key")

    for name in (
        "task_stack_in_sync_states",
        "git_merge_bases_by_instance",
        "git_trunk_timelines_by_instance",
        "git_ref_states_by_instance",
        "repo_executor_leases",
    ):
        if name in tables:
            op.drop_table(name)

    if "repositories" in tables:
        cols = {c["name"] for c in inspector.get_columns("repositories")}
        try:
            op.drop_constraint(
                "uq_repositories_workspace_repo_id", "repositories", type_="unique"
            )
        except Exception:
            pass
        with op.batch_alter_table("repositories") as batch:
            if "repo_id" in cols:
                batch.drop_column("repo_id")
            if "workspace_id" in cols:
                batch.drop_column("workspace_id")
