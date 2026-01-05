"""Add agent semantic fields to agent_sessions.

Revision ID: 0016_agent_session_semantics
Revises: 0015_remove_db_agent_construct
Create Date: 2026-01-05
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0016_agent_session_semantics"
down_revision = "0015_remove_db_agent_construct"
branch_labels = None
depends_on = None


def _json_type() -> sa.JSON:
    return sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def _dialect_server_default_json(conn, value: str) -> sa.TextClause:
    if conn.dialect.name == "postgresql":
        return sa.text(f"'{value}'::jsonb")
    return sa.text(f"'{value}'")


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = {str(col.get("name")) for col in inspector.get_columns("agent_sessions")}

    if "agent_capabilities" not in columns:
        op.add_column(
            "agent_sessions",
            sa.Column(
                "agent_capabilities",
                _json_type(),
                nullable=False,
                server_default=_dialect_server_default_json(
                    conn,
                    '{"can_detect_ready_for_input":false,'
                    '"can_detect_turn_complete":false,'
                    '"can_send_text":true,'
                    '"can_interrupt":false,'
                    '"can_receive_notifications":false,'
                    '"can_resume_by_id":false,'
                    '"can_continue_in_cwd":false,'
                    '"can_stream_semantic_events":false}',
                ),
            ),
        )

    if "agent_status" not in columns:
        op.add_column(
            "agent_sessions",
            sa.Column(
                "agent_status",
                _json_type(),
                nullable=False,
                server_default=_dialect_server_default_json(
                    conn, '{"turn_state":"unknown","detail":null}'
                ),
            ),
        )

    if "external_session_ref" not in columns:
        op.add_column(
            "agent_sessions",
            sa.Column(
                "external_session_ref",
                _json_type(),
                nullable=False,
                server_default=_dialect_server_default_json(conn, '{"type":"none"}'),
            ),
        )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = {str(col.get("name")) for col in inspector.get_columns("agent_sessions")}

    if "external_session_ref" in columns:
        op.drop_column("agent_sessions", "external_session_ref")
    if "agent_status" in columns:
        op.drop_column("agent_sessions", "agent_status")
    if "agent_capabilities" in columns:
        op.drop_column("agent_sessions", "agent_capabilities")

