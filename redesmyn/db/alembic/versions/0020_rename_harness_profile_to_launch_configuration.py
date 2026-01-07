"""Rename HarnessProfile to LaunchConfiguration.

The early agent orchestration work introduced a `harness_profiles` registry table
and referenced it from `agent_sessions` via `harness_profile_id`, with a per-run
snapshot stored in `resolved_profile`.

As the product terminology has stabilized, this data is more accurately a launch
configuration: argv/env/working-dir (plus optional bootstrap hints). This
migration renames the table and session fields accordingly.
"""

from __future__ import annotations

from collections.abc import Iterable

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine.interfaces import ReflectedColumn, ReflectedIndex

revision = "0020_rename_harness_profile_to_launch_configuration"
down_revision = "0019_agent_kind_selection"
branch_labels = None
depends_on = None


def _json_type() -> sa.JSON:
    return sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


CAPABILITIES_DEFAULT = (
    '{"can_detect_ready_for_input":false,'
    '"can_detect_turn_complete":false,'
    '"can_send_text":true,'
    '"can_interrupt":false,'
    '"can_receive_notifications":false,'
    '"can_resume_by_id":false,'
    '"can_continue_in_cwd":false,'
    '"can_stream_semantic_events":false}'
)

SEMANTIC_STATUS_DEFAULT = '{"turn_state":"unknown","detail":null}'

EXTERNAL_SESSION_DEFAULT = '{"type":"none"}'


def _server_default_json_literal(conn, value: str) -> str:
    if conn.dialect.name == "postgresql":
        return f"'{value}'::jsonb"
    return f"'{value}'"


def _index_names(indexes: Iterable[ReflectedIndex]) -> set[str]:
    names: set[str] = set()
    for index in indexes:
        name = index.get("name")
        if isinstance(name, str):
            names.add(name)
    return names


def _column_names(columns: Iterable[ReflectedColumn]) -> set[str]:
    names: set[str] = set()
    for column in columns:
        name = column.get("name")
        if isinstance(name, str):
            names.add(name)
    return names


def upgrade() -> None:
    conn = op.get_bind()
    inspector = inspect(conn)

    # SQLite batch migrations create `_alembic_tmp_*` tables; if a previous run was
    # interrupted, they can be left behind and break subsequent upgrades.
    if conn.dialect.name == "sqlite" and inspector.has_table(
        "_alembic_tmp_agent_sessions"
    ):
        op.drop_table("_alembic_tmp_agent_sessions")

    # Postgres: keep the enum values but rename the underlying type for clarity.
    if conn.dialect.name == "postgresql":
        harness_profile_enum_exists = bool(
            conn.execute(
                sa.text("SELECT 1 FROM pg_type WHERE typname = :name LIMIT 1"),
                {"name": "harness_profile_source"},
            ).scalar()
        )
        launch_configuration_enum_exists = bool(
            conn.execute(
                sa.text("SELECT 1 FROM pg_type WHERE typname = :name LIMIT 1"),
                {"name": "launch_configuration_source"},
            ).scalar()
        )
        if harness_profile_enum_exists and not launch_configuration_enum_exists:
            op.execute(
                "ALTER TYPE harness_profile_source RENAME TO launch_configuration_source"
            )

    has_harness_profiles = inspector.has_table("harness_profiles")
    has_launch_configurations = inspector.has_table("launch_configurations")
    if has_harness_profiles and has_launch_configurations:
        raise RuntimeError(
            "Cannot migrate: both harness_profiles and launch_configurations exist"
        )
    if has_harness_profiles and not has_launch_configurations:
        op.rename_table("harness_profiles", "launch_configurations")
    elif not has_harness_profiles and not has_launch_configurations:
        raise RuntimeError(
            "Cannot migrate: neither harness_profiles nor launch_configurations exist"
        )

    agent_session_columns = _column_names(inspector.get_columns("agent_sessions"))
    agent_session_indexes = _index_names(inspector.get_indexes("agent_sessions"))
    needs_index_rename = (
        "ix_agent_sessions_harness_profile_id" in agent_session_indexes
        and "ix_agent_sessions_launch_configuration_id" not in agent_session_indexes
    )

    with op.batch_alter_table("agent_sessions") as batch:
        if "harness_profile_id" in agent_session_columns:
            batch.alter_column(
                "harness_profile_id",
                new_column_name="launch_configuration_id",
                existing_type=sa.String(),
                existing_nullable=True,
            )
        if "resolved_profile" in agent_session_columns:
            batch.alter_column(
                "resolved_profile",
                new_column_name="resolved_launch_configuration",
                existing_nullable=True,
            )

        if conn.dialect.name == "sqlite":
            if "agent_capabilities" in agent_session_columns:
                batch.alter_column(
                    "agent_capabilities",
                    existing_type=_json_type(),
                    existing_nullable=False,
                    server_default=_server_default_json_literal(
                        conn, CAPABILITIES_DEFAULT
                    ),
                )
            if "agent_semantic_status" in agent_session_columns:
                batch.alter_column(
                    "agent_semantic_status",
                    existing_type=_json_type(),
                    existing_nullable=False,
                    server_default=_server_default_json_literal(
                        conn, SEMANTIC_STATUS_DEFAULT
                    ),
                )
            if "external_session_ref" in agent_session_columns:
                batch.alter_column(
                    "external_session_ref",
                    existing_type=_json_type(),
                    existing_nullable=False,
                    server_default=_server_default_json_literal(
                        conn, EXTERNAL_SESSION_DEFAULT
                    ),
                )

    # Work around an Alembic/SQLite batch-alter edge case where creating an index
    # on a column being renamed in the same batch can raise a KeyError.
    if needs_index_rename:
        op.drop_index(
            "ix_agent_sessions_harness_profile_id", table_name="agent_sessions"
        )
        op.create_index(
            "ix_agent_sessions_launch_configuration_id",
            "agent_sessions",
            ["launch_configuration_id"],
        )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = inspect(conn)

    if conn.dialect.name == "sqlite" and inspector.has_table(
        "_alembic_tmp_agent_sessions"
    ):
        op.drop_table("_alembic_tmp_agent_sessions")

    agent_session_columns = _column_names(inspector.get_columns("agent_sessions"))
    agent_session_indexes = _index_names(inspector.get_indexes("agent_sessions"))
    needs_index_rename = (
        "ix_agent_sessions_launch_configuration_id" in agent_session_indexes
        and "ix_agent_sessions_harness_profile_id" not in agent_session_indexes
    )

    with op.batch_alter_table("agent_sessions") as batch:
        if "launch_configuration_id" in agent_session_columns:
            batch.alter_column(
                "launch_configuration_id",
                new_column_name="harness_profile_id",
                existing_type=sa.String(),
                existing_nullable=True,
            )
        if "resolved_launch_configuration" in agent_session_columns:
            batch.alter_column(
                "resolved_launch_configuration",
                new_column_name="resolved_profile",
                existing_nullable=True,
            )

    if needs_index_rename:
        op.drop_index(
            "ix_agent_sessions_launch_configuration_id", table_name="agent_sessions"
        )
        op.create_index(
            "ix_agent_sessions_harness_profile_id",
            "agent_sessions",
            ["harness_profile_id"],
        )

    has_harness_profiles = inspector.has_table("harness_profiles")
    has_launch_configurations = inspector.has_table("launch_configurations")
    if has_harness_profiles and has_launch_configurations:
        raise RuntimeError(
            "Cannot migrate: both harness_profiles and launch_configurations exist"
        )
    if has_launch_configurations and not has_harness_profiles:
        op.rename_table("launch_configurations", "harness_profiles")

    if conn.dialect.name == "postgresql":
        harness_profile_enum_exists = bool(
            conn.execute(
                sa.text("SELECT 1 FROM pg_type WHERE typname = :name LIMIT 1"),
                {"name": "harness_profile_source"},
            ).scalar()
        )
        launch_configuration_enum_exists = bool(
            conn.execute(
                sa.text("SELECT 1 FROM pg_type WHERE typname = :name LIMIT 1"),
                {"name": "launch_configuration_source"},
            ).scalar()
        )
        if launch_configuration_enum_exists and not harness_profile_enum_exists:
            op.execute(
                "ALTER TYPE launch_configuration_source RENAME TO harness_profile_source"
            )
