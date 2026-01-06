"""Persist agent kind selection and resolved kind on agent_sessions.

This supports best-effort inference (Auto) plus explicit user override
(Generic/Codex/Claude Code) and keeps restarts deterministic.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0019_agent_kind_selection"
down_revision = "0018_fix_agent_session_json_defaults"
branch_labels = None
depends_on = None


AGENT_KIND_VALUES = ("generic", "codex", "claude_code")
AGENT_KIND_SELECTION_VALUES = ("auto", "generic", "codex", "claude_code")


def _enum(values: tuple[str, ...], *, name: str) -> sa.Enum:
    return sa.Enum(*values, name=name, native_enum=False)


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = {str(col.get("name")) for col in inspector.get_columns("agent_sessions")}

    if "agent_kind_selection" not in columns:
        op.add_column(
            "agent_sessions",
            sa.Column(
                "agent_kind_selection",
                _enum(AGENT_KIND_SELECTION_VALUES, name="agent_kind_selection"),
                nullable=False,
                server_default=sa.text("'auto'"),
            ),
        )

    if "agent_kind" not in columns:
        op.add_column(
            "agent_sessions",
            sa.Column(
                "agent_kind",
                _enum(AGENT_KIND_VALUES, name="agent_kind"),
                nullable=False,
                server_default=sa.text("'generic'"),
            ),
        )

    # Best-effort backfill from persisted external resume handles.
    if conn.dialect.name == "postgresql":
        op.execute(
            sa.text(
                "UPDATE agent_sessions "
                "SET agent_kind = 'codex' "
                "WHERE external_session_ref->>'type' = 'codex_thread'"
            )
        )
        op.execute(
            sa.text(
                "UPDATE agent_sessions "
                "SET agent_kind = 'claude_code' "
                "WHERE external_session_ref->>'type' = 'claude_session'"
            )
        )
    elif conn.dialect.name == "sqlite":
        op.execute(
            sa.text(
                "UPDATE agent_sessions "
                "SET agent_kind = 'codex' "
                "WHERE json_extract(external_session_ref, '$.type') = 'codex_thread'"
            )
        )
        op.execute(
            sa.text(
                "UPDATE agent_sessions "
                "SET agent_kind = 'claude_code' "
                "WHERE json_extract(external_session_ref, '$.type') = 'claude_session'"
            )
        )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = {str(col.get("name")) for col in inspector.get_columns("agent_sessions")}

    if "agent_kind" in columns:
        op.drop_column("agent_sessions", "agent_kind")
    if "agent_kind_selection" in columns:
        op.drop_column("agent_sessions", "agent_kind_selection")
