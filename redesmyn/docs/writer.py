from __future__ import annotations

from typing import Any

import yaml

from redesmyn.docs.markdown import MarkdownSectionError, _FENCE_RE, _HEADING_RE


SYNC_START = "<!-- rn:sync:start -->"
SYNC_END = "<!-- rn:sync:end -->"


def dump_yaml(data: dict[str, Any]) -> str:
    text = yaml.safe_dump(
        data,
        sort_keys=False,
        default_flow_style=False,
        width=88,
    )
    return text.rstrip() + "\n"


def _find_heading_line(lines: list[str], *, heading: str) -> int | None:
    target = heading.strip().lower()
    for idx, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m is None:
            continue
        if m.group("title").strip().lower() == target:
            return idx
    return None


def upsert_metadata_yaml(markdown: str, *, yaml_data: dict[str, Any]) -> str:
    lines = markdown.splitlines()
    heading_idx = _find_heading_line(lines, heading="Metadata")

    if heading_idx is None:
        insert_at = 0
        for idx, line in enumerate(lines):
            if line.startswith("# "):
                insert_at = idx + 1
                break
        block = [
            "## Metadata",
            "",
            "```yaml",
            dump_yaml(yaml_data).rstrip("\n"),
            "```",
            "",
        ]
        lines[insert_at:insert_at] = block
        return "\n".join(lines).rstrip() + "\n"

    fence_idx: int | None = None
    fence: str | None = None
    for idx in range(heading_idx + 1, len(lines)):
        line = lines[idx]
        if not line.strip():
            continue
        if _HEADING_RE.match(line) is not None:
            break
        m = _FENCE_RE.match(line)
        if m is None:
            continue
        fence_idx = idx
        fence = m.group("fence")
        break

    if fence_idx is None or fence is None:
        raise MarkdownSectionError("Missing fenced yaml block under '## Metadata'")

    end_idx: int | None = None
    for idx in range(fence_idx + 1, len(lines)):
        if lines[idx].strip() == fence:
            end_idx = idx
            break
    if end_idx is None:
        raise MarkdownSectionError("Unterminated fenced block under '## Metadata'")

    yaml_lines = dump_yaml(yaml_data).rstrip("\n").splitlines()
    lines[fence_idx + 1 : end_idx] = yaml_lines
    return "\n".join(lines).rstrip() + "\n"


def upsert_synced_section(markdown: str, *, title: str, content: str) -> str:
    body = content.rstrip() + "\n"
    start = markdown.find(SYNC_START)
    end = markdown.find(SYNC_END)

    if start != -1 and end != -1 and start < end:
        start_end = start + len(SYNC_START)
        return markdown[:start_end] + "\n" + body + markdown[end:]

    section = (
        f"\n## {title}\n\n{SYNC_START}\n{body}{SYNC_END}\n"
    )
    return markdown.rstrip() + section
