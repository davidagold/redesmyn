---
epic: gpui
branch:
  suggested: rn/gpui/T-76-codex-app-server-protocol-conformance
rn:
  node:
    branch: rn/gpui/T-76-codex-app-server-protocol-conformance
  parent: T-68
---

# T-76 Codex app-server protocol conformance harness (fixtures + drift detection) (Domain 4)

## Problem

T-68 intentionally implements a **lightweight** Codex app-server runner:

- minimal `Content-Length` framing + JSON-RPC envelope code, and
- small, purpose-built serde models for the v2 method subset we use.

This keeps builds fast and avoids coupling Redesmyn’s daemon runtime to the full Codex codebase.

However, it introduces a risk: **protocol drift**. Codex can evolve v2 method names, params, and
diff item shapes in ways that compile cleanly for us but break at runtime.

## Goal

Add an opt-in conformance harness that:

- validates our framing + serde models against a pinned Codex app-server protocol reference,
- provides deterministic fixtures for regression tests,
- and can be run in CI or locally without impacting normal dev build times.

## Requirements

### 1) Keep normal builds lightweight

- Do **not** make `codex-app-server-protocol` a required dependency for the default Rust workspace build.
- Options (pick one):
  - add a small conformance crate that depends on `codex-app-server-protocol`, or
  - add it as a `dev-dependency` behind a Cargo feature used only by conformance tests.

### 2) Pinned reference version

- Pin the Codex reference revision explicitly (no floating main).
- Document the update process (e.g. “bump pin → refresh fixtures → run conformance suite”).

### 3) Deterministic fixtures

Create a fixture set covering the minimum v2 surface we rely on, including:

- `initialize` request/response,
- `newConversation` response including `conversationId` (treated as our external session id),
- `userMessage` request with and without `resume`,
- `updateConversation` diffs:
  - `newConversation` + `newTurn`,
  - `newMessage` + `updateMessage` delta path,
  - `turnCompleted` variants (completed / failed / canceled),
- server → client request: `conversationId`,
- server → client request: `loginWithChatGPT` (with and without URL),
- server → client notifications: `runCommand`, `proposalToEditFile`.

Fixtures should be stable across runs and stored in-repo.

### 4) Conformance tests

Add tests that:

- decode fixtures with both:
  - our lightweight models, and
  - the pinned Codex protocol crate models (as the reference),
  and assert key fields match (ids, turn ids, message content, “done” semantics).
- validate our outbound JSON-RPC messages can be parsed by the reference models.

### 5) CI integration (non-blocking to start)

- Add a command to run the conformance suite locally (e.g. `cargo test -p ... --features ...`).
- CI can start as an optional/manual job and later become required once stable.

## Acceptance criteria

- A developer can run the conformance suite and get actionable failures when Codex protocol shapes drift.
- Updating the pinned Codex reference revision is a deterministic, documented process.
- Normal dev builds remain fast (no always-on dependency on Codex’s protocol crate).

