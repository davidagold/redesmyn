from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import yaml


class MarkdownSectionError(ValueError):
    pass


_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\\s+(?P<title>[^#].*?)\\s*$")
_FENCE_RE = re.compile(r"^(?P<fence>`{3,}|~{3,})\\s*(?P<lang>[A-Za-z0-9_-]+)?\\s*$")


@dataclass(frozen=True, slots=True)
class FencedBlock:
    fence: str
    lang: str | None
    content: str


def _is_heading(line: str) -> bool:
    return _HEADING_RE.match(line) is not None


def extract_fenced_block_after_heading(
    markdown: str,
    *,
    heading: str,
    allowed_langs: set[str] | None = None,
) -> FencedBlock:
    lines = markdown.splitlines()
    target = heading.strip().lower()
    heading_index: int | None = None

    for idx, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m is None:
            continue
        title = m.group("title").strip().lower()
        if title == target:
            heading_index = idx
            break

    if heading_index is None:
        raise MarkdownSectionError(f"Missing heading: {heading!r}")

    fence_index: int | None = None
    fence: str | None = None
    lang: str | None = None
    for idx in range(heading_index + 1, len(lines)):
        line = lines[idx]
        if not line.strip():
            continue
        if _is_heading(line):
            break
        m = _FENCE_RE.match(line)
        if m is None:
            continue
        fence_index = idx
        fence = m.group("fence")
        lang_raw = m.group("lang")
        lang = lang_raw.lower() if lang_raw else None
        if allowed_langs is not None and lang is not None and lang not in allowed_langs:
            raise MarkdownSectionError(
                f"Unexpected fenced block language {lang!r} under {heading!r}; "
                f"expected one of: {sorted(allowed_langs)}"
            )
        break

    if fence_index is None or fence is None:
        raise MarkdownSectionError(f"Missing fenced block under heading: {heading!r}")

    content_start = fence_index + 1
    content_end: int | None = None
    for idx in range(content_start, len(lines)):
        if lines[idx].strip() == fence:
            content_end = idx
            break

    if content_end is None:
        raise MarkdownSectionError(
            f"Unterminated fenced block under heading: {heading!r}"
        )

    content = "\n".join(lines[content_start:content_end]).rstrip() + "\n"
    return FencedBlock(fence=fence, lang=lang, content=content)


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

