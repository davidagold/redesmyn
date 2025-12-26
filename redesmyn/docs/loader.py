from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from redesmyn.docs.markdown import (
    MarkdownSectionError,
    extract_fenced_block_after_heading,
    parse_yaml_block,
)
from redesmyn.docs.metadata import EpicMetadata, TaskMetadata


class DocLoadError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class EpicDoc:
    path: Path
    markdown: str
    metadata: EpicMetadata


@dataclass(frozen=True, slots=True)
class TaskDoc:
    path: Path
    markdown: str
    metadata: TaskMetadata
    title: str | None


def _extract_title(markdown: str) -> str | None:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line.removeprefix("# ").strip() or None
    return None


def load_epic_doc(path: Path) -> EpicDoc:
    try:
        markdown = path.read_text(encoding="utf-8")
    except OSError as e:
        raise DocLoadError(f"Failed to read {path}: {e}") from e

    try:
        block = extract_fenced_block_after_heading(
            markdown, heading="Metadata", allowed_langs={"yaml", "yml"}
        )
        data = parse_yaml_block(block.content)
        metadata = EpicMetadata.model_validate(data)
    except (MarkdownSectionError, ValidationError) as e:
        raise DocLoadError(f"Invalid epic doc metadata in {path}: {e}") from e

    return EpicDoc(path=path, markdown=markdown, metadata=metadata)


def load_task_doc(path: Path) -> TaskDoc:
    try:
        markdown = path.read_text(encoding="utf-8")
    except OSError as e:
        raise DocLoadError(f"Failed to read {path}: {e}") from e

    try:
        block = extract_fenced_block_after_heading(
            markdown, heading="Metadata", allowed_langs={"yaml", "yml"}
        )
        data = parse_yaml_block(block.content)
        metadata = TaskMetadata.model_validate(data)
    except (MarkdownSectionError, ValidationError) as e:
        raise DocLoadError(f"Invalid task doc metadata in {path}: {e}") from e

    return TaskDoc(
        path=path, markdown=markdown, metadata=metadata, title=_extract_title(markdown)
    )
