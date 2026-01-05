"""Store Linear issue identifier on tasks.

Revision ID: 0012_task_linear_identifier
Revises: 0011_create_agent_sessions
Create Date: 2026-01-05
"""

from __future__ import annotations

import re

from alembic import op
import sqlalchemy as sa
import yaml

revision = "0012_task_linear_identifier"
down_revision = "0011_create_agent_sessions"
branch_labels = None
depends_on = None

_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>[^#].*?)\s*$")
_FENCE_RE = re.compile(r"^(?P<fence>`{3,}|~{3,})\s*(?P<lang>[A-Za-z0-9_-]+)?\s*$")
_TITLE_IDENTIFIER_RE = re.compile(r"^(?P<identifier>[A-Z][A-Z0-9]+-\d+)\b")


def _extract_linear_identifier(
    *, markdown: str | None, title: str | None
) -> str | None:
    if markdown:
        lines = markdown.splitlines()
        heading_index: int | None = None
        for idx, line in enumerate(lines):
            m = _HEADING_RE.match(line)
            if m is None:
                continue
            if m.group("title").strip().lower() == "metadata":
                heading_index = idx
                break

        if heading_index is not None:
            fence: str | None = None
            lang: str | None = None
            fence_index: int | None = None
            for idx in range(heading_index + 1, len(lines)):
                line = lines[idx]
                if not line.strip():
                    continue
                if _HEADING_RE.match(line) is not None:
                    break
                m = _FENCE_RE.match(line)
                if m is None:
                    continue
                fence_index = idx
                fence = m.group("fence")
                lang_raw = m.group("lang")
                lang = lang_raw.lower() if lang_raw else None
                break

            if fence_index is not None and fence is not None:
                if lang is None or lang in {"yaml", "yml"}:
                    content_lines: list[str] = []
                    for idx in range(fence_index + 1, len(lines)):
                        if lines[idx].strip() == fence:
                            break
                        content_lines.append(lines[idx])
                    if content_lines:
                        try:
                            data = yaml.safe_load("\n".join(content_lines))
                        except yaml.YAMLError:
                            data = None
                        if isinstance(data, dict):
                            linear = data.get("linear")
                            if isinstance(linear, dict):
                                raw = linear.get("identifier")
                                if isinstance(raw, str):
                                    candidate = raw.split("#", 1)[0].strip()
                                    candidate = candidate.strip("'\"")
                                    if candidate:
                                        return candidate

    if title:
        m = _TITLE_IDENTIFIER_RE.match(title.strip())
        if m is not None:
            return m.group("identifier")

    return None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "tasks" not in tables:
        return

    cols = {c["name"] for c in inspector.get_columns("tasks")}
    if "linear_identifier" not in cols:
        with op.batch_alter_table("tasks") as batch:
            batch.add_column(sa.Column("linear_identifier", sa.String(), nullable=True))

    rows = conn.execute(
        sa.text("SELECT id, title, body FROM tasks WHERE linear_identifier IS NULL")
    ).fetchall()
    for task_id, title, body in rows:
        identifier = _extract_linear_identifier(markdown=body, title=title)
        if not identifier:
            continue
        conn.execute(
            sa.text("UPDATE tasks SET linear_identifier = :identifier WHERE id = :id"),
            {"identifier": identifier, "id": task_id},
        )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "tasks" not in tables:
        return

    cols = {c["name"] for c in inspector.get_columns("tasks")}
    if "linear_identifier" not in cols:
        return

    with op.batch_alter_table("tasks") as batch:
        batch.drop_column("linear_identifier")
