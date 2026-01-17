from __future__ import annotations

from typing import Any

import yaml


class MarkdownSectionError(ValueError):
    pass


_FRONTMATTER_DELIMS = {"---", "..."}


def parse_yaml_block(block: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(block)
    except yaml.YAMLError as e:
        raise MarkdownSectionError(f"Invalid YAML: {e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise MarkdownSectionError("YAML metadata must be a mapping/object")
    return data


def extract_yaml_frontmatter(markdown: str) -> str:
    text = markdown
    if text.startswith("\ufeff"):
        text = text.removeprefix("\ufeff")

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise MarkdownSectionError(
            "Missing YAML frontmatter (expected '---' on the first line)"
        )

    end_idx: int | None = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() in _FRONTMATTER_DELIMS:
            end_idx = idx
            break

    if end_idx is None:
        raise MarkdownSectionError(
            "Unterminated YAML frontmatter (missing closing '---')"
        )

    return "\n".join(lines[1:end_idx]).rstrip() + "\n"


def split_yaml_frontmatter_document(markdown: str) -> tuple[dict[str, Any], str]:
    text = markdown
    if text.startswith("\ufeff"):
        text = text.removeprefix("\ufeff")

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    end_idx: int | None = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() in _FRONTMATTER_DELIMS:
            end_idx = idx
            break
    if end_idx is None:
        raise MarkdownSectionError(
            "Unterminated YAML frontmatter (missing closing '---')"
        )

    frontmatter_raw = "\n".join(lines[1:end_idx]).rstrip() + "\n"
    data = parse_yaml_block(frontmatter_raw)
    body = "\n".join(lines[end_idx + 1 :]).lstrip("\n").rstrip() + "\n"
    return data, body
