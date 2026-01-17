from __future__ import annotations

from typing import Any

import yaml

from redesmyn.docs.markdown import MarkdownSectionError, parse_yaml_block

_FRONTMATTER_DELIMS = {"---", "..."}


def dump_yaml(data: dict[str, Any]) -> str:
    text = yaml.safe_dump(
        data,
        sort_keys=False,
        default_flow_style=False,
        width=88,
    )
    return text.rstrip() + "\n"


def upsert_rn_metadata(markdown: str, *, rn_data: dict[str, Any]) -> str:
    bom = "\ufeff" if markdown.startswith("\ufeff") else ""
    text = markdown.removeprefix(bom)
    lines = text.splitlines()

    frontmatter_dict: dict[str, Any] = {}
    end_idx: int | None = None
    if lines and lines[0].strip() == "---":
        for idx in range(1, len(lines)):
            if lines[idx].strip() in _FRONTMATTER_DELIMS:
                end_idx = idx
                break
        if end_idx is None:
            raise MarkdownSectionError(
                "Unterminated YAML frontmatter (missing closing '---')"
            )
        existing_raw = "\n".join(lines[1:end_idx]).rstrip() + "\n"
        frontmatter_dict = parse_yaml_block(existing_raw)

    frontmatter_dict = dict(frontmatter_dict)
    frontmatter_dict["rn"] = dict(rn_data)

    yaml_block = dump_yaml(frontmatter_dict).rstrip("\n")
    frontmatter = f"---\n{yaml_block}\n---\n"

    if end_idx is not None:
        rest = "\n".join(lines[end_idx + 1 :]).lstrip("\n")
        if rest:
            return (bom + frontmatter + "\n" + rest).rstrip() + "\n"
        return bom + frontmatter

    rest = "\n".join(lines).lstrip("\n")
    if rest:
        return (bom + frontmatter + "\n" + rest).rstrip() + "\n"
    return bom + frontmatter
