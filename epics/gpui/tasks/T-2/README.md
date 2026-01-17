---
epic: gpui
branch:
  suggested: rn/gpui/T-2-ulid-newtypes
rn:
  parent: null
---

# T-2 ULID + newtypes everywhere (`redesmyn_ids`) (Domain 0)

## Problem

The current codebase uses a mix of integer IDs, strings, and ad-hoc identifiers across boundaries (DB, API, websocket, UI).

For the Rust port, we want:

- strong typing by default,
- faster lookups and indexing,
- consistent identity semantics across daemon/control-plane/UI,
- fewer “wrong ID in the wrong place” bugs.

## Goal

Introduce a `redesmyn_ids` crate providing ULID-backed newtypes for all primary identifiers in the Rust port, and standardize how IDs are represented:

- in memory (newtypes),
- on the wire (JSON/protobuf),
- in SQLite storage (`sqlx`).

## Requirements

### 1) ID types

Create ULID-backed newtypes for (at minimum):

- `WorkspaceId`
- `RepoId`
- `EpicId`
- `TaskId`
- `RunId`
- `HostId` (or `HostKey` if we want to preserve that term)
- `CommandId`
- `EventId`

Notes:

- Prefer `#[repr(transparent)]` newtypes over a single “Id<T>” generic to keep debugging and trait derivations straightforward.
- Implement `Copy` where reasonable (ULID is 16 bytes; evaluate tradeoffs, but prefer ergonomics).

### 2) Wire encoding

- JSON: canonical ULID string.
- Protobuf: 16 raw bytes (or a fixed 128-bit field) where feasible.

We keep string encoding available for diagnostics.

### 3) Storage encoding (SQLite via `sqlx`)

Store ULIDs as `BLOB(16)` for performance/index size, with helpers to render readable ULIDs in logs/UI.

### 4) Ergonomics

- Implement `Display`/`FromStr` (good error messages).
- Implement serde serialize/deserialize.
- Add tests:
  - parse/format roundtrips
  - serde roundtrip
  - sqlx encode/decode roundtrip (SQLite)

## Acceptance criteria

- `redesmyn_ids` is a dependency root for most crates; newtypes are used instead of raw strings/ints in new Rust code.
- DB and wire encoding strategies are explicitly tested and documented.
- The crate is tiny, fast to compile, and has no heavy dependencies outside what’s needed (`ulid`, `serde`, minimal `sqlx` support).

