---
id: T-2
stacked_on: T-1
must_land_after: []
node:
  branch: rn/tests-v0/T-2-db-migrations
---

# T-2 DB/migrations + schema invariants

## Goal

Add integration-level tests that prove the database schema is healthy and that migrations remain compatible with runtime expectations.

## Motivation

We have already hit cases where the runtime ORM expectations drifted from the SQLite schema in the field (e.g., legacy `NOT NULL` columns causing 500s at runtime). v0 should make this kind of drift hard to reintroduce.

## Requirements

### 1) Migrations apply cleanly

- A fresh database should upgrade through the full Alembic chain without errors.
- Where feasible, also test upgrading from at least one older revision (if we have a stable reference point).

### 2) Minimal runtime invariants

Validate critical invariants that affect user-visible behavior, e.g.:

- columns that must have defaults
- fields required by API routes
- constraints that protect correctness (where present)

This should be pragmatic: we don’t need exhaustive schema tests, but we should protect the failure modes we’ve seen.

## Suggested approach

- Use the scenario DB fixture(s) from T-1 to create a temp db, run migrations, then validate:
  - tables exist
  - critical columns exist + are nullable/non-nullable as expected
  - inserting key records works with current code paths

## Acceptance Criteria

- `uv run pytest -k migrations` passes.
- At least one test explicitly covers “legacy schema drift” class of issues (to prevent reintroduction).
