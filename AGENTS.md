# Redesmyn Agent Notes

- Prefer clean, thoughtfully organized commits when asked to commit (keep history manageable and easy to rebase/split).
- For any Python usage that relies on project packages/scripts, use `uv` (e.g. `uv run …`, `uv sync --dev`); avoid `pip install`.
- We are still designing v0 of the database (no users/testing DB yet) — do not add migrations.
- Domain model source of truth is the SQLAlchemy ORM in `redesmyn/db/models.py` (no separate dataclass layer).
- JSON data/payload columns use `JSON_TYPE` (SQLite JSON + Postgres JSONB). Validate with the Pydantic models in the same file (`CommandPayload`, `EventData`).
- Prefer model names without the `Model` suffix (e.g. `CommandPayload`, not `CommandPayloadModel`).
- Enums use `StrEnum` with PascalCase member names and snake_case values; SQLAlchemy enums must store values (not names).
