"""Fix invalid JSON defaults added in 0017 for SQLite.

SQLite's JSON columns are stored as TEXT; the server_default text used in 0017
was constructed via `sa.text(...)`, which treats `:false` / `:null` sequences as
bind params and produced defaults like `{"key"NULL}`. Those strings are not
valid JSON and can crash SQLAlchemy on row load (json.loads).

This migration:
- backfills any invalid JSON values to known-good defaults
- updates server defaults to literal SQL (no bind parsing)
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0018_fix_agent_session_json_defaults"
down_revision = "0017_agent_session_semantics"
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


def _server_default_json_literal(conn, value: str) -> sa.ColumnElement:
    if conn.dialect.name == "postgresql":
        return sa.literal_column(f"'{value}'::jsonb")
    return sa.literal_column(f"'{value}'")


def upgrade() -> None:
    conn = op.get_bind()

    # Repair any bad values written by 0017's broken SQLite defaults.
    if conn.dialect.name == "sqlite":
        op.execute(
            sa.text(
                "UPDATE agent_sessions "
                "SET agent_capabilities = :value "
                "WHERE agent_capabilities IS NULL "
                "OR json_valid(agent_capabilities) = 0"
            ).bindparams(value=CAPABILITIES_DEFAULT)
        )
        op.execute(
            sa.text(
                "UPDATE agent_sessions "
                "SET agent_semantic_status = :value "
                "WHERE agent_semantic_status IS NULL "
                "OR json_valid(agent_semantic_status) = 0"
            ).bindparams(value=SEMANTIC_STATUS_DEFAULT)
        )

    default_capabilities = _server_default_json_literal(conn, CAPABILITIES_DEFAULT)
    default_semantic_status = _server_default_json_literal(
        conn, SEMANTIC_STATUS_DEFAULT
    )

    if conn.dialect.name == "sqlite":
        with op.batch_alter_table("agent_sessions") as batch:
            batch.alter_column(
                "agent_capabilities",
                existing_type=_json_type(),
                existing_nullable=False,
                server_default=default_capabilities,
            )
            batch.alter_column(
                "agent_semantic_status",
                existing_type=_json_type(),
                existing_nullable=False,
                server_default=default_semantic_status,
            )
    else:
        op.alter_column(
            "agent_sessions",
            "agent_capabilities",
            existing_type=_json_type(),
            existing_nullable=False,
            server_default=default_capabilities,
        )
        op.alter_column(
            "agent_sessions",
            "agent_semantic_status",
            existing_type=_json_type(),
            existing_nullable=False,
            server_default=default_semantic_status,
        )


def downgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name == "sqlite":
        with op.batch_alter_table("agent_sessions") as batch:
            batch.alter_column(
                "agent_capabilities",
                existing_type=_json_type(),
                existing_nullable=False,
                server_default=None,
            )
            batch.alter_column(
                "agent_semantic_status",
                existing_type=_json_type(),
                existing_nullable=False,
                server_default=None,
            )
    else:
        op.alter_column(
            "agent_sessions",
            "agent_capabilities",
            existing_type=_json_type(),
            existing_nullable=False,
            server_default=None,
        )
        op.alter_column(
            "agent_sessions",
            "agent_semantic_status",
            existing_type=_json_type(),
            existing_nullable=False,
            server_default=None,
        )
