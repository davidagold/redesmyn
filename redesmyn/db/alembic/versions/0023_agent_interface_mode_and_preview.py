"""Persist agent interface mode and assistant message preview on agent_sessions.

- agent_interface_mode: whether the session is intended to run with structured signals
  (e.g. Codex JSONL / Claude stream-json) vs interactive heuristic mode.
- agent_preview: bounded preview state derived from semantic events (v0: assistant message).
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0023_agent_interface_mode_and_preview"
down_revision = "0022_linear_epic_defaults_label_name"
branch_labels = None
depends_on = None


AGENT_INTERFACE_MODE_VALUES = ("interactive", "structured")

AGENT_PREVIEW_DEFAULT = (
    '{"last_assistant_message_preview":null,'
    '"last_assistant_message_at":null,'
    '"last_message_turn_id":null}'
)

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


def _enum(values: tuple[str, ...], *, name: str) -> sa.Enum:
    return sa.Enum(*values, name=name, native_enum=False)


def _json_type() -> sa.JSON:
    return sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def _server_default_json_literal(conn, value: str) -> str:
    if conn.dialect.name == "postgresql":
        return f"'{value}'::jsonb"
    return f"'{value}'"


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = {str(col.get("name")) for col in inspector.get_columns("agent_sessions")}

    if "agent_interface_mode" not in columns:
        op.add_column(
            "agent_sessions",
            sa.Column(
                "agent_interface_mode",
                _enum(AGENT_INTERFACE_MODE_VALUES, name="agent_interface_mode"),
                nullable=False,
                server_default=sa.text("'interactive'"),
            ),
        )

    if "agent_preview" not in columns:
        op.add_column(
            "agent_sessions",
            sa.Column(
                "agent_preview",
                _json_type(),
                nullable=False,
                server_default=_server_default_json_literal(
                    conn, AGENT_PREVIEW_DEFAULT
                ),
            ),
        )

    if conn.dialect.name == "sqlite":
        op.execute(
            sa.text(
                "UPDATE agent_sessions "
                "SET agent_capabilities = :value "
                "WHERE agent_capabilities IS NULL OR json_valid(agent_capabilities) = 0"
            ).bindparams(value=CAPABILITIES_DEFAULT)
        )
        op.execute(
            sa.text(
                "UPDATE agent_sessions "
                "SET agent_semantic_status = :value "
                "WHERE agent_semantic_status IS NULL OR json_valid(agent_semantic_status) = 0"
            ).bindparams(value=SEMANTIC_STATUS_DEFAULT)
        )
        op.execute(
            sa.text(
                "UPDATE agent_sessions "
                "SET agent_preview = :value "
                "WHERE agent_preview IS NULL OR json_valid(agent_preview) = 0"
            ).bindparams(value=AGENT_PREVIEW_DEFAULT)
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
    inspector = sa.inspect(conn)
    columns = {str(col.get("name")) for col in inspector.get_columns("agent_sessions")}

    if "agent_preview" in columns:
        op.drop_column("agent_sessions", "agent_preview")
    if "agent_interface_mode" in columns:
        op.drop_column("agent_sessions", "agent_interface_mode")
