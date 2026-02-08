---
epic: gpui
branch:
  suggested: rn/gpui/T-68-codex-app-server-runner
rn:
  node:
    branch: rn/gpui/T-68-codex-app-server-runner
  parent: T-39
  after:
    - T-32
    - T-33
    - T-35
    - T-41
---

# T-68 Codex app-server runner (daemon) (JSON-RPC over stdio; protocol v2) (Domain 4)

## Problem

We want “app-server” agents to be first-class in the GPUI/Rust port:

- long-lived server process (no per-turn respawn),
- request/response control surface,
- and a structured event stream that can drive the native session viewer.

Codex already offers an **app-server** mode. If we don’t implement it in this epic, we’ll:

- lock ourselves into the exec-based transport as the only structured path,
- and delay the “session viewer as a wrapper around a session” direction for exec-style agents.

Implementers may not have internet access during development, so this ticket includes the concrete protocol surface we expect to target.

## Goal

Implement a daemon-side **Codex app-server runner** that:

- launches `codex app-server` in the correct repo/worktree/sandbox context,
- speaks Codex app-server protocol v2 (JSON-RPC over stdio with `Content-Length` framing),
- converts Codex “session diff” notifications (Codex calls them `updateConversation`) into Redesmyn `SessionEvent` emissions (T-14),
- persists the external session id (Codex calls it `conversationId`) as the resumable session handle,
- and integrates with control plane “send message” semantics (T-41) and session persistence (T-40).

## Requirements

### 1) Runtime taxonomy + integration seam

- Use the shared taxonomy/types from T-32 (`AgentProvider`, `AgentRuntimeKind`, `ExternalSessionRef`, etc.).
- This runner is `AgentProvider::Codex` + `AgentRuntimeKind::AppServer`.
- It must be consumable through the same control-plane command surface (T-41) as StructuredExec Codex (T-37); app-server must not require new
  “special-cased” UI or client APIs beyond capability checks.

### 2) Transport framing (stdio)

Codex app-server protocol is JSON-RPC over stdio with LSP-style framing:

- Each frame is:
  - ASCII headers, including `Content-Length: <bytes>\r\n`,
  - terminated by `\r\n`,
  - followed by exactly `<bytes>` of UTF-8 JSON.
- No “JSONL per line” assumptions.

Implementation guidance:

- Implement the minimal framing + message models in-tree (keep it small and tested).
- Track upstream protocol drift via a separate conformance harness ticket (T-76) rather than pulling the full Codex protocol crate into every build.

### 3) JSON-RPC method surface (protocol v2, minimum viable subset)

We target the v2 method names and message shapes as defined in Codex’s `codex-app-server-protocol`:

**App → Server requests**

- `initialize` (must include `protocolVersion` and auth params; see below)
- `newConversation`
- `userMessage` (supports `resume: Option<String>` to resume an existing session)
- `cancel`
- `commandExecutionApproval`
- `fileChangeApproval`

**App → Server notifications**

- `initialized`
- `exit`

**Server → App notifications**

- `updateConversation` (primary structured stream: an ordered list of “diff items”)
- `runCommand` (server asks client to run a shell command)
- `proposalToEditFile` (server proposes a file edit)
- `appendToLog` (append a line to a log stream)
- `rerender`
- `shutdown`

**Server → App requests**

- `conversationId` (the client responds with the active session id)
- `loginWithChatGPT` (OAuth bootstrap; client opens URL in the user’s browser)

Auth basics:

- `initialize` includes `auth: { accessToken?: string, refreshToken?: string, accountId?: string }` (exact naming per protocol crate).
- For v0 of this runner, it is acceptable to rely on Codex’s existing local auth store *if present* and surface a user-actionable error when login is required.

### 4) Mapping Codex app-server events → Redesmyn session events

Codex’s primary stream is the `updateConversation` notification:

- `updateConversation.params.diff: Vec<SessionDiff>`
- Each diff item is tagged (examples include):
  - `newConversation` (includes `conversationId`, which we treat as the external session id)
  - `newTurn`
  - `newMessage` / `updateMessage`
  - `newToolCall` / `updateToolCall`
  - `newToolResult` / `updateToolResult`
  - `newPatch` / `updatePatch`
  - `newLog` / `updateLog`

Define a deterministic mapping to Redesmyn `SessionEvent` (T-14) such that:

- The session’s `ExternalSessionRef` is persisted as soon as the external session id is known.
- “session == conversation; turns are events” holds:
  - `newTurn` emits `agent.turn_started`,
  - turn completion/failure is emitted deterministically (based on diff sequence).
- All non-chunk/delta emissions required for native history are persisted (same rule as exec-based Codex).

Delivery path (important):

- The runner emits `SessionEvent` updates to the control plane as `DaemonMessage::SessionEventBatch` frames; persistence happens in the control plane (T-40).

Where Codex uses “update*” diff items, prefer to emit:

- a durable “final” event when the item stabilizes (e.g., a message is complete),
- and optionally emit live-only delta events for UI polish (not persisted; not required for history).

### 5) “Send message” + interrupt semantics

Runner must support the control plane semantics (T-41):

- Start a new conversation when no external handle exists.
- Resume an existing session when `ExternalSessionRef` is present.
- Provide an “interrupt/cancel” path:
  - use `cancel` request where appropriate,
  - always surface visible lifecycle updates (no silent failure).

### 6) Testability (no real Codex required)

Provide deterministic tests without requiring Codex installed:

- Implement a small fake app-server process fixture that:
  - speaks the framed JSON-RPC transport,
  - accepts `initialize`/`newConversation`/`userMessage`,
  - emits `updateConversation` diffs in a controlled way (including the external session id),
  - and supports `cancel`.

Acceptance tests should cover:

- session id capture + persistence,
  - at least one user message → assistant message turn,
  - interrupt/cancel,
  - and robust parsing of framing under chunked reads.

## Acceptance criteria

- Daemon can run a mock Codex app-server and emit Redesmyn `SessionEvent`s suitable for SessionView history.
- External session id is captured and persisted promptly and used for subsequent resumed turns.
- Runner integrates with control-plane “send message” semantics (conflict handling stays in control plane; runner exposes the right primitive operations).

- Observability: new code paths include deliberate `tracing` spans/logs via `redesmyn_logging` (key lifecycle + errors; avoid noisy per-request/per-tick spam).

## Dependencies / sequencing

- Depends on:
  - Agent taxonomy + turn-intent primitives (T-32),
  - Codex structured parsing strategy (T-33) *or* an explicit adapter for app-server diffs,
  - Exec-session supervisor conventions (T-35) for process lifecycle patterns (even if the app-server runner uses a separate supervisor),
  - Control plane agent commands semantics (T-41),
  - Session event contract (T-14).
- Builds on:
  - app-server runtime skeleton (T-39).

## Reference implementation (external; for protocol orientation only)

These are mentioned here so the implementing agent can align terminology with Codex, but the ticket body above is intended to be sufficient even without web access:

- Codex app-server protocol crate: `codex-rs/app-server-protocol` (v2 definitions; framing + message types).
