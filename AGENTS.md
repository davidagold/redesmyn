# Redesmyn Agent Notes

- Prefer clean, thoughtfully organized, change-sized commits when asked to commit; avoid mixing unrelated concerns (e.g. keep layout vs. markdown vs. UX changes separate) so history stays easy to rebase/split.
- For any Python usage that relies on project packages/scripts, use `uv` (e.g. `uv run …`, `uv sync`); avoid `pip install`.
- Domain model source of truth is the SQLAlchemy ORM in `redesmyn/db/models.py` (no separate dataclass layer).
- JSON data/payload columns use `JSON_TYPE` (SQLite JSON + Postgres JSONB). Validate with the Pydantic models in the same file (`CommandData`, `EventData`).
- Prefer model names without the `Model` suffix (e.g. `CommandData`, not `CommandDataModel`).
- Enums use `StrEnum` with PascalCase member names and snake_case values; SQLAlchemy enums must store values (not names).
