"""Merge Node into Task (single graph primitive).

Revision ID: 0002_merge_node_into_task
Revises: 0001_baseline
Create Date: 2025-12-31
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0002_merge_node_into_task"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None


def _sqlite_supports_json(conn) -> bool:
    try:
        conn.execute(sa.text("SELECT json('{\"ok\": true}')"))
    except Exception:
        return False
    return True


def upgrade() -> None:
    conn = op.get_bind()

    with op.batch_alter_table("tasks") as batch:
        batch.add_column(sa.Column("branch_name", sa.String(), nullable=True))
        batch.add_column(sa.Column("parent_task_id", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("agent_id", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("worktree_path", sa.String(), nullable=True))
        batch.add_column(sa.Column("github_pr_id", sa.String(), nullable=True))

    node_rows = list(
        conn.execute(
            sa.text(
                "SELECT id, parent_node_id, branch_name, agent_id, worktree_path, "
                "primary_task_id, github_pr_id "
                "FROM nodes ORDER BY id"
            )
        ).mappings()
    )

    node_to_task_id: dict[int, int] = {}
    for row in node_rows:
        node_id = int(row["id"])
        primary_task_id = row["primary_task_id"]
        task_id: int | None = (
            int(primary_task_id) if primary_task_id is not None else None
        )
        if task_id is None:
            task_id = conn.execute(
                sa.text(
                    "SELECT id FROM tasks WHERE node_id = :node_id ORDER BY id LIMIT 1"
                ),
                {"node_id": node_id},
            ).scalar()
            task_id = int(task_id) if task_id is not None else None
        if task_id is not None:
            node_to_task_id[node_id] = task_id

    for row in node_rows:
        node_id = int(row["id"])
        task_id = node_to_task_id.get(node_id)
        if task_id is None:
            continue

        parent_node_id = row["parent_node_id"]
        parent_task_id: int | None = None
        if parent_node_id is not None:
            parent_task_id = node_to_task_id.get(int(parent_node_id))

        conn.execute(
            sa.text(
                "UPDATE tasks SET "
                "branch_name = :branch_name, "
                "parent_task_id = :parent_task_id, "
                "agent_id = :agent_id, "
                "worktree_path = :worktree_path, "
                "github_pr_id = :github_pr_id "
                "WHERE id = :task_id"
            ),
            {
                "branch_name": row["branch_name"],
                "parent_task_id": parent_task_id,
                "agent_id": row["agent_id"],
                "worktree_path": row["worktree_path"],
                "github_pr_id": row["github_pr_id"],
                "task_id": task_id,
            },
        )

    with op.batch_alter_table("commands") as batch:
        batch.add_column(sa.Column("target_task_id", sa.Integer(), nullable=True))

    command_rows = list(
        conn.execute(sa.text("SELECT id, target_node_id FROM commands")).mappings()
    )
    for row in command_rows:
        cmd_id = int(row["id"])
        target_node_id = row["target_node_id"]
        if target_node_id is None:
            continue
        target_task_id = node_to_task_id.get(int(target_node_id))
        if target_task_id is None:
            continue
        conn.execute(
            sa.text("UPDATE commands SET target_task_id = :tid WHERE id = :id"),
            {"tid": target_task_id, "id": cmd_id},
        )

    if conn.dialect.name == "sqlite" and _sqlite_supports_json(conn):
        # Translate legacy event payloads so subscriptions/filters can be task-based.
        # (Best-effort; unknown payload shapes are left as-is.)
        for node_id, task_id in node_to_task_id.items():
            conn.execute(
                sa.text(
                    "UPDATE events "
                    "SET data = json_set(json_remove(data, '$.node_id'), '$.task_id', :task_id) "
                    "WHERE json_extract(data, '$.node_id') = :node_id"
                ),
                {"node_id": node_id, "task_id": task_id},
            )

        conn.execute(
            sa.text(
                "UPDATE events SET event_type = 'task.agent_set' WHERE event_type = 'node.agent_set'"
            )
        )

    with op.batch_alter_table("tasks") as batch:
        batch.create_foreign_key(
            "fk_tasks_parent_task_id", "tasks", ["parent_task_id"], ["id"]
        )
        batch.create_foreign_key("fk_tasks_agent_id", "agents", ["agent_id"], ["id"])
        batch.create_index("ix_tasks_parent_task_id", ["parent_task_id"])
        batch.create_index("ix_tasks_agent_id", ["agent_id"])
        batch.create_unique_constraint(
            "uq_tasks_epic_branch", ["epic_id", "branch_name"]
        )
        batch.drop_index("ix_tasks_node_id")
        batch.drop_column("node_id")

    with op.batch_alter_table("commands") as batch:
        batch.create_index("ix_commands_target_task_id", ["target_task_id"])
        batch.drop_index("ix_commands_target_node_id")
        batch.drop_column("target_node_id")

    op.drop_table("nodes")


def downgrade() -> None:
    conn = op.get_bind()

    op.create_table(
        "nodes",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("epic_id", sa.Integer(), nullable=False),
        sa.Column("branch_name", sa.String(), nullable=False),
        sa.Column("parent_node_id", sa.Integer(), nullable=True),
        sa.Column("agent_id", sa.Integer(), nullable=True),
        sa.Column("worktree_path", sa.String(), nullable=True),
        sa.Column("primary_task_id", sa.Integer(), nullable=True),
        sa.Column("github_pr_id", sa.String(), nullable=True),
        sa.Column("linear_issue_id", sa.String(), nullable=True),
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
        sa.ForeignKeyConstraint(["agent_id"], ["agents.id"]),
        sa.ForeignKeyConstraint(["epic_id"], ["epics.id"]),
        sa.ForeignKeyConstraint(["parent_node_id"], ["nodes.id"]),
        sa.ForeignKeyConstraint(["primary_task_id"], ["tasks.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("epic_id", "branch_name", name="uq_nodes_epic_branch"),
    )
    op.create_index("ix_nodes_agent_id", "nodes", ["agent_id"])
    op.create_index("ix_nodes_epic_id", "nodes", ["epic_id"])
    op.create_index("ix_nodes_parent_node_id", "nodes", ["parent_node_id"])
    op.create_index("ix_nodes_primary_task_id", "nodes", ["primary_task_id"])

    with op.batch_alter_table("tasks") as batch:
        batch.add_column(sa.Column("node_id", sa.Integer(), nullable=True))
        batch.create_index("ix_tasks_node_id", ["node_id"])
        batch.create_foreign_key("fk_tasks_node_id", "nodes", ["node_id"], ["id"])

    # Recreate nodes from tasks that have branch metadata.
    task_rows = list(
        conn.execute(
            sa.text(
                "SELECT id, epic_id, branch_name, parent_task_id, agent_id, worktree_path, github_pr_id "
                "FROM tasks WHERE branch_name IS NOT NULL ORDER BY id"
            )
        ).mappings()
    )
    task_to_node_id: dict[int, int] = {}
    for row in task_rows:
        result = conn.execute(
            sa.text(
                "INSERT INTO nodes (epic_id, branch_name, parent_node_id, agent_id, worktree_path, primary_task_id, github_pr_id) "
                "VALUES (:epic_id, :branch_name, NULL, :agent_id, :worktree_path, :primary_task_id, :github_pr_id)"
            ),
            {
                "epic_id": row["epic_id"],
                "branch_name": row["branch_name"],
                "agent_id": row["agent_id"],
                "worktree_path": row["worktree_path"],
                "primary_task_id": row["id"],
                "github_pr_id": row["github_pr_id"],
            },
        )
        node_id = int(result.lastrowid)  # type: ignore[attr-defined]
        task_to_node_id[int(row["id"])] = node_id

    for row in task_rows:
        task_id = int(row["id"])
        parent_task_id = row["parent_task_id"]
        if parent_task_id is None:
            continue
        node_id = task_to_node_id.get(task_id)
        parent_node_id = task_to_node_id.get(int(parent_task_id))
        if node_id is None or parent_node_id is None:
            continue
        conn.execute(
            sa.text("UPDATE nodes SET parent_node_id = :pid WHERE id = :id"),
            {"pid": parent_node_id, "id": node_id},
        )

    for task_id, node_id in task_to_node_id.items():
        conn.execute(
            sa.text("UPDATE tasks SET node_id = :node_id WHERE id = :task_id"),
            {"node_id": node_id, "task_id": task_id},
        )

    with op.batch_alter_table("commands") as batch:
        batch.add_column(sa.Column("target_node_id", sa.Integer(), nullable=True))
        batch.create_index("ix_commands_target_node_id", ["target_node_id"])
        batch.create_foreign_key(
            "fk_commands_target_node_id", "nodes", ["target_node_id"], ["id"]
        )

    cmd_rows = list(
        conn.execute(sa.text("SELECT id, target_task_id FROM commands")).mappings()
    )
    for row in cmd_rows:
        cmd_id = int(row["id"])
        target_task_id = row["target_task_id"]
        if target_task_id is None:
            continue
        node_id = task_to_node_id.get(int(target_task_id))
        if node_id is None:
            continue
        conn.execute(
            sa.text("UPDATE commands SET target_node_id = :nid WHERE id = :id"),
            {"nid": node_id, "id": cmd_id},
        )

    with op.batch_alter_table("commands") as batch:
        batch.drop_index("ix_commands_target_task_id")
        batch.drop_column("target_task_id")

    if conn.dialect.name == "sqlite" and _sqlite_supports_json(conn):
        # Best-effort reversal of the payload key and event type rename.
        conn.execute(
            sa.text(
                "UPDATE events "
                "SET data = json_set(json_remove(data, '$.task_id'), '$.node_id', json_extract(data, '$.task_id')) "
                "WHERE json_extract(data, '$.task_id') IS NOT NULL"
            )
        )
        conn.execute(
            sa.text(
                "UPDATE events SET event_type = 'node.agent_set' WHERE event_type = 'task.agent_set'"
            )
        )

    with op.batch_alter_table("tasks") as batch:
        batch.drop_constraint("fk_tasks_parent_task_id", type_="foreignkey")
        batch.drop_constraint("fk_tasks_agent_id", type_="foreignkey")
        batch.drop_index("ix_tasks_parent_task_id")
        batch.drop_index("ix_tasks_agent_id")
        batch.drop_constraint("uq_tasks_epic_branch", type_="unique")
        batch.drop_column("branch_name")
        batch.drop_column("parent_task_id")
        batch.drop_column("agent_id")
        batch.drop_column("worktree_path")
        batch.drop_column("github_pr_id")
