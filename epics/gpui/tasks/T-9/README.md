---
epic: gpui
branch:
  suggested: rn/gpui/T-9-protocol-envelope
rn:
  parent: T-2
---

# T-9 Protocol envelope + versioning + scopes (Domain 1)

## Problem

We are building a system that is “distributed by design” even when embedded:

- A headless **control plane** (authoritative state, event log + projections, APIs).
- A **daemon** (repo executor) that owns git/worktrees/agents/telemetry.
- Multiple **clients** (GPUI desktop UI, `rn` CLI) that operate as control-plane clients.

Even when the desktop app embeds both control plane and daemon, the application logic must not assume colocation.

Without an explicit, stable, versioned message envelope and scope model, we risk:

- brittle transports,
- accidental coupling across boundaries,
- and a protocol that cannot evolve safely.

## Goal

Define the canonical protocol envelope and core compatibility rules used by:

- daemon ↔ control plane streams (in-proc and remote),
- client ↔ control plane API transports (local Unix socket and future remote),
- JSON diagnostic tooling and Protobuf wire transports.

This ticket is about **types and semantics**, not about implementing full control-plane or daemon behavior.

## Requirements

### 1) Envelope fields (canonical)

Define an envelope with these fields and semantics:

- `protocol_major: u16`
- `protocol_minor: u16`
- `msg_id: MsgId` (ULID; idempotency + dedupe key)
- `sent_at: Timestamp` (monotonic ordering is not assumed)
- `scope: Option<Scope>` (see below)
- `correlation_id: Option<MsgId>` (request/response correlation; optional)
- `trace_id: Option<TraceId>` (for tracing across hops; optional)

Notes:

- `msg_id` is required and must be globally unique; receivers must dedupe best-effort by `msg_id`.
- `correlation_id` is used to correlate responses/acks to initiating requests when needed; it is not a substitute for domain IDs like `command_id` or `run_id`.
- `sent_at` is informational; ordering-sensitive flows must use explicit sequence numbers in the payload when needed.

### 2) Scope model (future-proof, minimal now)

Define routing scope as an enum/oneof, carried in the envelope as `scope: Option<Scope>`:

- `Repo { workspace_id: WorkspaceId, repo_id: RepoId }`

Rules:

- Repo-scoped messages must include `Some(Scope::Repo { ... })`.
- Non-repo-scoped messages must omit scope (`None`) and are routed at the connection level.
- We intentionally leave room for future scope kinds (e.g. workspace-level) without breaking the envelope.

Persistence note (Domain 2):

- When scope is stored in DB tables as `(workspace_id, repo_id)`, treat it as a single logical
  identity and prefer composite foreign keys (vs separate FKs) so “mismatched pairs” cannot be
  persisted (see T-17).

### 3) Versioning rules

- Major mismatch: reject with a structured error (and close the connection).
- Minor mismatch: accept if possible; unknown fields must not break parsing.
- Additive fields are allowed; removing or changing semantics requires major bump.

Define where version negotiation occurs:

- daemon stream: in the initial handshake / hello exchange.
- client API: during connection establishment (first request/response).

### 4) Encoding rules (JSON vs Protobuf)

Define canonical encoding rules:

- IDs (ULID/newtypes):
  - JSON: ULID string
  - Protobuf: 16-byte representation
- Timestamp:
  - JSON: RFC3339 string
  - Protobuf: `google.protobuf.Timestamp` (or equivalent)

### 5) Error envelope (cross-boundary)

Define a minimal structured error type that can appear as:

- a message payload (protocol-level error), and/or
- a response status (client API).

Minimum fields:

- `category` (small stable enum: invalid_request, not_found, conflict, unauthorized, unavailable, internal)
- `message` (user-actionable text; no stack traces)
- `detail` (optional structured data for debugging/UX)

## Acceptance criteria

- The envelope + scope model is written down in a canonical location (in this ticket and as Rust types in `redesmyn_protocol` once implemented).
- All later protocol messages in this epic reference and reuse this envelope (no bespoke headers).
- Versioning and encoding rules are explicit enough that two implementers can build compatible code without guessing.

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on `epics/gpui/tasks/T-2/README.md` (ULID/newtypes).
- Unblocks the rest of Domain 1 (schema/codegen, daemon protocol, client API, tooling).

## Reference implementation (today; envelope-ish semantics)

- WS protocol (Python today):
  - `redesmyn/ws_protocol.py` (JSON message schema; hello/ping/resync-ish patterns).
  - `redesmyn/event_stream.py` + `redesmyn/ws_runtime.py` (runtime behavior for `/v1/ws`).
- WS protocol (TS today):
  - `dashboard/src/hooks/useEventStream.ts` (event union + resync/hello message shapes).
