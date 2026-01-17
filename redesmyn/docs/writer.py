from __future__ import annotations

from typing import Any

import yaml

from redesmyn.docs.markdown import MarkdownSectionError


SYNC_START = "<!-- rn:sync:start -->"
SYNC_END = "<!-- rn:sync:end -->"
_FRONTMATTER_DELIMS = {"---", "..."}


def dump_yaml(data: dict[str, Any]) -> str:
    text = yaml.safe_dump(
        data,
        sort_keys=False,
        default_flow_style=False,
        width=88,
    )
    return text.rstrip() + "\n"


def upsert_metadata_yaml(markdown: str, *, yaml_data: dict[str, Any]) -> str:
    bom = "\ufeff" if markdown.startswith("\ufeff") else ""
    text = markdown.removeprefix(bom)
    lines = text.splitlines()

    yaml_block = dump_yaml(yaml_data).rstrip("\n")
    frontmatter = f"---\n{yaml_block}\n---\n"

    if lines and lines[0].strip() == "---":
        end_idx: int | None = None
        for idx in range(1, len(lines)):
            if lines[idx].strip() in _FRONTMATTER_DELIMS:
                end_idx = idx
                break
        if end_idx is None:
            raise MarkdownSectionError(
                "Unterminated YAML frontmatter (missing closing '---')"
            )

        rest = "\n".join(lines[end_idx + 1 :]).lstrip("\n")
        if rest:
            return (bom + frontmatter + "\n" + rest).rstrip() + "\n"
        return bom + frontmatter

    rest = "\n".join(lines).lstrip("\n")
    if rest:
        return (bom + frontmatter + "\n" + rest).rstrip() + "\n"
    return bom + frontmatter


def upsert_synced_section(markdown: str, *, title: str, content: str) -> str:
    body = content.rstrip() + "\n"
    start = markdown.find(SYNC_START)
    end = markdown.find(SYNC_END)

    if start != -1 and end != -1 and start < end:
        start_end = start + len(SYNC_START)
        return markdown[:start_end] + "\n" + body + markdown[end:]

    section = f"\n## {title}\n\n{SYNC_START}\n{body}{SYNC_END}\n"
    return markdown.rstrip() + section
