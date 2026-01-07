"""Rename HarnessProfile to LaunchConfiguration.

The early agent orchestration work introduced a `harness_profiles` registry table
and referenced it from `agent_sessions` via `harness_profile_id`, with a per-run
snapshot stored in `resolved_profile`.

As the product terminology has stabilized, this data is more accurately a launch
configuration: argv/env/working-dir (plus optional bootstrap hints). This
migration renames the table and session fields accordingly.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0020_rename_harness_profile_to_launch_configuration"
down_revision = "0019_agent_kind_selection"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # Postgres: keep the enum values but rename the underlying type for clarity.
    if conn.dialect.name == "postgresql":
        op.execute(
            "ALTER TYPE harness_profile_source RENAME TO launch_configuration_source"
        )

    op.rename_table("harness_profiles", "launch_configurations")

    with op.batch_alter_table("agent_sessions") as batch:
        batch.drop_index("ix_agent_sessions_harness_profile_id")
        batch.alter_column(
            "harness_profile_id",
            new_column_name="launch_configuration_id",
            existing_type=sa.String(),
            existing_nullable=True,
        )
        batch.alter_column(
            "resolved_profile",
            new_column_name="resolved_launch_configuration",
            existing_nullable=True,
        )
        batch.create_index(
            "ix_agent_sessions_launch_configuration_id",
            ["launch_configuration_id"],
        )


def downgrade() -> None:
    conn = op.get_bind()

    with op.batch_alter_table("agent_sessions") as batch:
        batch.drop_index("ix_agent_sessions_launch_configuration_id")
        batch.alter_column(
            "launch_configuration_id",
            new_column_name="harness_profile_id",
            existing_type=sa.String(),
            existing_nullable=True,
        )
        batch.alter_column(
            "resolved_launch_configuration",
            new_column_name="resolved_profile",
            existing_nullable=True,
        )
        batch.create_index(
            "ix_agent_sessions_harness_profile_id",
            ["harness_profile_id"],
        )

    op.rename_table("launch_configurations", "harness_profiles")

    if conn.dialect.name == "postgresql":
        op.execute(
            "ALTER TYPE launch_configuration_source RENAME TO harness_profile_source"
        )
