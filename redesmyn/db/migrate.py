from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.util import CommandError


def _repo_root_from_db_path(db_path: Path) -> Path:
    # db_path is typically: <repo>/.redesmyn/redesmyn.sqlite3
    return db_path.parent.parent


def alembic_config_for_db(db_path: Path) -> Config:
    repo_root = _repo_root_from_db_path(db_path)
    cfg = Config(str(repo_root / "alembic.ini"))
    # When called programmatically (e.g., via `rn`), we don't want Alembic's
    # default logging config to spam INFO lines on every invocation.
    cfg.attributes["configure_logger"] = False
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


def upgrade_to_head(*, db_path: Path) -> None:
    command.upgrade(alembic_config_for_db(db_path), "head")


def stamp_revision(*, db_path: Path, revision: str) -> None:
    command.stamp(alembic_config_for_db(db_path), revision)


def head_revision(*, db_path: Path) -> str:
    cfg = alembic_config_for_db(db_path)
    script = ScriptDirectory.from_config(cfg)
    try:
        head = script.get_current_head()
    except CommandError as e:
        heads = script.get_heads()
        joined = ", ".join(heads) if heads else "(none)"
        hint = (
            "Resolve by merging the heads with Alembic (e.g. "
            f'`uv run alembic -c alembic.ini merge -m "merge heads" {" ".join(heads)}`) '
            "and committing the new migration."
            if heads
            else "Resolve by adding an Alembic merge revision."
        )
        raise RuntimeError(
            f"Alembic has multiple heads (migration conflict likely): {joined}. {hint}"
        ) from e
    if head is None:
        raise RuntimeError("Alembic script directory has no head revision")
    return head
