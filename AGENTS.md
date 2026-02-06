# Redesmyn Agent Notes

## Workflow

- Work one task per branch/worktree, and commit as you go (avoid large uncommitted states).
- Prefer clean, thoughtfully organized, change-sized commits; avoid mixing unrelated concerns (e.g. keep layout vs. markdown vs. UX changes separate) so history stays easy to rebase/split.
- Keep task branches stacked: each task branch should be based on the tip of its parent task branch (`parent`), rebasing as needed.
- For any Python usage that relies on project packages/scripts, use `uv` (e.g. `uv run …`, `uv sync`); avoid `pip install`.
- Strive for self-documenting code via clear names and sensible factoring; if logic/settings are non-obvious or easy to break, add a brief comment explaining why.
- Keep the codebase well-typed: prefer typed data models (Pydantic, enums, `Literal`/union types) over unstructured `str`/`dict` payloads unless there is a compelling necessity.
- Leverage existing types to keep code simple (avoid overly defensive type-guards / `getattr`-style access when strong typing is available).
- Be thoughtful about design and architecture: favor simple, maintainable, composable building blocks and avoid very long functions; when a function has multiple distinct steps, split into helpers whose names document the flow and act as single sources of truth.
- When writing tests, follow `tests/README.md`.
- Keep Alembic migrations linear on `main`: before merging a branch that adds migrations, rebase/renumber so the new revision points at the current single head (avoid adding Alembic merge revisions except as a last resort).

## Data Model Conventions

- Domain model source of truth is the SQLAlchemy ORM in `redesmyn/db/models.py` (no separate dataclass layer).
- JSON data/payload columns use `JSON_TYPE` (SQLite JSON + Postgres JSONB). Validate with the Pydantic models in the same file (`CommandData`, `EventData`).
- Prefer model names without the `Model` suffix (e.g. `CommandData`, not `CommandDataModel`).
- Enums use `StrEnum` with PascalCase member names and snake_case values; SQLAlchemy enums must store values (not names).

## Dashboard UI Conventions

- Avoid adding borders to every card/panel; too many lines makes the UI feel busy and distracts from the information. Prefer spacing and subtle rules/separators; reserve borders for elements that truly need to pop against their background.
- Avoid leaking palette ingredient names (e.g. Rose Pine “pine”) into public-facing theme tokens or UI code; prefer semantic theme properties (e.g. `info`, `success`) and reference those instead.
- Avoid inert property enumerations (e.g. “From/To” blocks that restate what the graph already shows). Prefer structured UI that leverages the graph (highlighting, selection states, breadcrumbs) and dedicate the Details panel to actionable content (contracts, messages, controls).
- Never initiate a user-visible action (API request, daemon command, git action) without an immediate and clearly visible “in progress” indication (no “silent” dead air after a click).
- No spinner wheels. Shimmering text is reserved only for LLM generation.
- Prefer calm-but-visible progress affordances: animated ellipses, subtle glow/pulse, or similar low-noise motion.
- Prevent accidental duplicate requests: disable the triggering control while in flight unless concurrent actions are explicitly safe.
- Keep progress indicators accessible (visible in light/dark, keyboard-safe, no focus traps); on error, keep messages actionable and preserve user input when possible (e.g. don’t drop drafts).
- Avoid repeating the same fact in multiple places (e.g. callout + inline label + status text). Pick one primary surface to carry the message, then keep secondary surfaces quiet unless they add new information.
- Let the main UI element convey intent directly through state, placement, and wording; reserve helper copy/tooltips for brief clarification, not restating what is already visible.

## Projections vs. Live State

- If a value materially affects user expectations (e.g. approvals/sandbox), don’t display a default as if it were authoritative unless it has been confirmed via durable events or a live backend query.
- Use projections for fast initial UI, but reconcile against live state as soon as it’s reachable; if neither is available, show “Loading…” / “Unknown” rather than misleading defaults.

## Performance

- Treat per-frame render/layout work as a budget: avoid allocations and avoid building large element subtrees in hot paths (zoom, pan, animations).
- Prefer viewport culling first (don’t build offscreen node/card UI); graduate to full virtualization only when culling is insufficient.
- Bound and/or prune caches (LRU/FIFO + retain-by-reference) to prevent steady-state memory growth; never leave “append-only” UI caches unbounded.
- When an interaction triggers repeated relayout (expand/collapse, resize), keep transitional UI lightweight; avoid doing expensive parsing/shaping on every frame.
