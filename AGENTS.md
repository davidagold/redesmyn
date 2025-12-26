# Redesmyn Agent Notes

## Workflow

- When asked to commit, prefer clean, thoughtfully organized, change-sized commits; avoid mixing unrelated concerns (e.g. keep layout vs. markdown vs. UX changes separate) so history stays easy to rebase/split.
- For any Python usage that relies on project packages/scripts, use `uv` (e.g. `uv run …`, `uv sync`); avoid `pip install`.

## Data Model Conventions

- Domain model source of truth is the SQLAlchemy ORM in `redesmyn/db/models.py` (no separate dataclass layer).
- JSON data/payload columns use `JSON_TYPE` (SQLite JSON + Postgres JSONB). Validate with the Pydantic models in the same file (`CommandData`, `EventData`).
- Prefer model names without the `Model` suffix (e.g. `CommandData`, not `CommandDataModel`).
- Enums use `StrEnum` with PascalCase member names and snake_case values; SQLAlchemy enums must store values (not names).

## Dashboard UI Conventions

- Avoid adding borders to every card/panel; too many lines makes the UI feel busy and distracts from the information. Prefer spacing and subtle rules/separators; reserve borders for elements that truly need to pop against their background.
