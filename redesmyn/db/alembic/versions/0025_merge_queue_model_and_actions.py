"""Add merge queue model tables and durable action log.

Revision ID: 0025_merge_queue_model_and_actions
Revises: 0024_github_repo_association
Create Date: 2026-02-10
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0025_merge_queue_model_and_actions"
down_revision = "0024_github_repo_association"
branch_labels = None
depends_on = None


MERGE_QUEUE_ITEM_STATE_VALUES = (
    "draft",
    "ready",
    "gated",
    "mergeable",
    "merged",
    "blocked",
    "deferred",
)

MERGE_QUEUE_CONDUCTOR_DECISION_VALUES = (
    "pending",
    "approved",
    "approved_pending",
    "changes_requested",
    "rejected",
    "deferred",
)

MERGE_QUEUE_DEPENDENCY_KIND_VALUES = (
    "hard",
    "approval_pending",
    "follow_up",
)

MERGE_QUEUE_ACTION_AUTHORITY_VALUES = (
    "director",
    "conductor",
    "system",
)

MERGE_QUEUE_ACTION_TYPE_VALUES = (
    "enqueue",
    "candidate_updated",
    "state_updated",
    "dependency_added",
    "dependency_removed",
    "approve",
    "approve_pending",
    "request_changes",
    "reject",
    "defer",
    "requeue",
    "pause",
    "resume",
    "reorder",
    "mark_merged",
    "mark_blocked",
)


def _enum(values: tuple[str, ...], *, name: str) -> sa.Enum:
    return sa.Enum(*values, name=name, native_enum=False)


def _json_type() -> sa.JSON:
    return sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    op.create_table(
        "merge_queue_items",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("epic_id", sa.Integer(), nullable=False),
        sa.Column("task_id", sa.Integer(), nullable=True),
        sa.Column("candidate_ref", sa.String(), nullable=False),
        sa.Column(
            "state",
            _enum(MERGE_QUEUE_ITEM_STATE_VALUES, name="merge_queue_item_state"),
            nullable=False,
            server_default=sa.text("'draft'"),
        ),
        sa.Column(
            "conductor_decision",
            _enum(
                MERGE_QUEUE_CONDUCTOR_DECISION_VALUES,
                name="merge_queue_conductor_decision",
            ),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column(
            "order_index", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column("paused", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("blocked_reason", sa.Text(), nullable=True),
        sa.Column("deferred_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("merged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "candidate_ref <> ''",
            name="ck_merge_queue_items_candidate_ref",
        ),
        sa.ForeignKeyConstraint(["epic_id"], ["epics.id"]),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_merge_queue_items_epic_id"),
        "merge_queue_items",
        ["epic_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_merge_queue_items_task_id"),
        "merge_queue_items",
        ["task_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_merge_queue_items_state"),
        "merge_queue_items",
        ["state"],
        unique=False,
    )
    op.create_index(
        op.f("ix_merge_queue_items_conductor_decision"),
        "merge_queue_items",
        ["conductor_decision"],
        unique=False,
    )

    op.create_table(
        "merge_queue_dependencies",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("epic_id", sa.Integer(), nullable=False),
        sa.Column("queue_item_id", sa.Integer(), nullable=False),
        sa.Column("depends_on_item_id", sa.Integer(), nullable=False),
        sa.Column(
            "kind",
            _enum(
                MERGE_QUEUE_DEPENDENCY_KIND_VALUES,
                name="merge_queue_dependency_kind",
            ),
            nullable=False,
            server_default=sa.text("'hard'"),
        ),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("satisfied_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "queue_item_id <> depends_on_item_id",
            name="ck_merge_queue_dependencies_no_self_dependency",
        ),
        sa.ForeignKeyConstraint(["epic_id"], ["epics.id"]),
        sa.ForeignKeyConstraint(["queue_item_id"], ["merge_queue_items.id"]),
        sa.ForeignKeyConstraint(["depends_on_item_id"], ["merge_queue_items.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "queue_item_id",
            "depends_on_item_id",
            "kind",
            name="uq_merge_queue_dependencies_unique_edge",
        ),
    )
    op.create_index(
        op.f("ix_merge_queue_dependencies_epic_id"),
        "merge_queue_dependencies",
        ["epic_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_merge_queue_dependencies_queue_item_id"),
        "merge_queue_dependencies",
        ["queue_item_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_merge_queue_dependencies_depends_on_item_id"),
        "merge_queue_dependencies",
        ["depends_on_item_id"],
        unique=False,
    )

    op.create_table(
        "merge_queue_actions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("epic_id", sa.Integer(), nullable=False),
        sa.Column("queue_item_id", sa.Integer(), nullable=False),
        sa.Column(
            "authority",
            _enum(
                MERGE_QUEUE_ACTION_AUTHORITY_VALUES,
                name="merge_queue_action_authority",
            ),
            nullable=False,
        ),
        sa.Column(
            "action",
            _enum(
                MERGE_QUEUE_ACTION_TYPE_VALUES,
                name="merge_queue_action_type",
            ),
            nullable=False,
        ),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("data", _json_type(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column(
            "from_state",
            _enum(
                MERGE_QUEUE_ITEM_STATE_VALUES,
                name="merge_queue_action_from_state",
            ),
            nullable=True,
        ),
        sa.Column(
            "to_state",
            _enum(MERGE_QUEUE_ITEM_STATE_VALUES, name="merge_queue_action_to_state"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(["epic_id"], ["epics.id"]),
        sa.ForeignKeyConstraint(["queue_item_id"], ["merge_queue_items.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_merge_queue_actions_epic_id"),
        "merge_queue_actions",
        ["epic_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_merge_queue_actions_queue_item_id"),
        "merge_queue_actions",
        ["queue_item_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_merge_queue_actions_authority"),
        "merge_queue_actions",
        ["authority"],
        unique=False,
    )
    op.create_index(
        op.f("ix_merge_queue_actions_action"),
        "merge_queue_actions",
        ["action"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_merge_queue_actions_action"), table_name="merge_queue_actions"
    )
    op.drop_index(
        op.f("ix_merge_queue_actions_authority"), table_name="merge_queue_actions"
    )
    op.drop_index(
        op.f("ix_merge_queue_actions_queue_item_id"), table_name="merge_queue_actions"
    )
    op.drop_index(
        op.f("ix_merge_queue_actions_epic_id"), table_name="merge_queue_actions"
    )
    op.drop_table("merge_queue_actions")

    op.drop_index(
        op.f("ix_merge_queue_dependencies_depends_on_item_id"),
        table_name="merge_queue_dependencies",
    )
    op.drop_index(
        op.f("ix_merge_queue_dependencies_queue_item_id"),
        table_name="merge_queue_dependencies",
    )
    op.drop_index(
        op.f("ix_merge_queue_dependencies_epic_id"),
        table_name="merge_queue_dependencies",
    )
    op.drop_table("merge_queue_dependencies")

    op.drop_index(
        op.f("ix_merge_queue_items_conductor_decision"), table_name="merge_queue_items"
    )
    op.drop_index(op.f("ix_merge_queue_items_state"), table_name="merge_queue_items")
    op.drop_index(op.f("ix_merge_queue_items_task_id"), table_name="merge_queue_items")
    op.drop_index(op.f("ix_merge_queue_items_epic_id"), table_name="merge_queue_items")
    op.drop_table("merge_queue_items")
