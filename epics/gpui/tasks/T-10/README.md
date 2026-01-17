---
epic: gpui
branch:
  suggested: rn/gpui/T-10-protobuf-schema-codegen
rn:
  parent: T-9
---

# T-10 Protobuf schemas + codegen pipeline (Domain 1)

## Problem

We want a high-performance, maintainable protocol with:

- Protobuf as the default network codec,
- JSON as an opt-in diagnostic codec,
- and typed Rust message structures as the source of truth.

If codegen and schema ownership are unclear, we risk:

- ad-hoc message formats,
- fragile build steps,
- and incompatible “almost the same” message definitions across crates.

## Goal

Establish a clean Protobuf schema layout and Rust code generation pipeline that:

- is easy to run (no exotic tooling required),
- produces stable, versioned message types,
- and makes it hard to accidentally diverge between JSON and Protobuf representations.

## Requirements

### 1) Schema ownership + layout

- Store `.proto` files under `rust/proto/`.
- Organize by logical surface:
  - `envelope.proto` (shared envelope/scopes, per T-9)
  - `daemon.proto` (daemon ↔ control plane stream messages)
  - `client.proto` (client ↔ control plane API messages)
  - `artifacts.proto` (artifact refs; may start minimal)

### 2) Code generation approach

Use `prost` / `prost-build` for Rust generation.

Constraints:

- `cargo build` should work without requiring contributors to install extra tooling beyond Rust (optional tools like `buf` can be “nice to have”, not required).
- Generated code should not be checked in unless there is a compelling reason; prefer build-time generation.

### 3) Compatibility + “unknown fields”

Ensure the Protobuf design supports forward compatibility:

- avoid `required` fields,
- avoid semantic overload of field numbers,
- prefer additive changes,
- and define a policy for “unknown event types” (see T-11/T-14) so old clients can still display something meaningful.

### 4) JSON diagnostic representation

Define how to produce JSON for diagnostics:

- Either:
  - implement serde on our native Rust message structs, with explicit conversions to/from Protobuf structs, or
  - use a protobuf→json mapping with strict rules.

Preference for this epic:

- Native Rust structs are the primary API, and we provide:
  - Protobuf conversions for wire transport, and
  - serde JSON for debug tooling.

### 5) Tests

Add tests that assert:

- Protobuf encode/decode roundtrip for envelope + at least one message type.
- JSON encode/decode roundtrip for the same native message type.
- ULID and timestamp mapping follow the canonical rules from T-9.

## Acceptance criteria

- `.proto` files exist in the workspace with clear ownership and comments.
- `cargo test` can generate and compile the protocol bindings.
- A minimal set of message types can be serialized/deserialized via:
  - Protobuf (wire), and
  - JSON (diagnostic).

## Dependencies / sequencing

- Depends on `epics/gpui/tasks/T-9/README.md` (envelope + rules).
- Unblocks daemon/client protocol implementation and all transports/codecs.

## Reference implementation (today; codegen precedent)

- OpenAPI generation (Python today):
  - `scripts/export_openapi.py` (exports spec).
  - `openapi/openapi.json` (spec consumed by the dashboard).
- TS client generation (today):
  - `dashboard/package.json` (`api:update`, `api:gen` scripts).
  - `dashboard/src/api/v1.ts` (generated client types).
- Non-OpenAPI realtime protocol (today):
  - `redesmyn/ws_protocol.py` + `dashboard/src/hooks/useEventStream.ts` (hand-rolled JSON WS protocol).
