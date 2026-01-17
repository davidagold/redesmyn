---
rn:
  slug: gpui
  name: GPUI + Rust Port
  root_branch: gpui
  linear:
    project_id: null
---

# GPUI + Rust Port Epic: Control Doc (Canonical)

This file is the canonical “control doc” for the **GPUI + Rust Port** epic: intent, scope, invariants, architecture, and key decisions. Keep it current.

## 1) Vision

Redesmyn becomes a **standalone desktop application** built with **Rust + GPUI**.

The desktop app embeds:

- A **control plane** (persistent state, APIs, projections).
- A “modularly embedded” **daemon** (repo executor: git/worktrees/agents/telemetry).

Even when embedded in one desktop process, **the application logic must make no assumptions that the control plane and daemon are colocated**. The architecture must make a later transition to **remote daemons** as straightforward as possible.

The `rn` CLI remains **first-class** and becomes **very fast** (“zippy af”). The UI is still **graph-first** and must never initiate user-visible actions without immediate, clearly visible “in progress” feedback (no silent actions).

Supported OS: **macOS + Linux**. Windows is not required for this epic.

## 2) Principles (non-negotiable)

- **Separation by construction**: control plane code must not be able to reach repo filesystem operations except through a daemon/executor boundary.
- **Simple, explicit abstractions**: minimize magic; prefer small crates and clear ownership boundaries.
- **Strong typing**: prefer newtypes + enums and typed models; avoid “stringly typed” payloads.
- **Performance is a product feature**: design for high-throughput event streams, large diffs, and smooth UI interactions.
- **Observability without noise**: structured tracing by default; rich debugging hooks when needed.
- **No silent actions**: every mutation has immediate visible progress; prevent accidental duplicate requests.
- **Maintainability first**: idiomatic Rust, small modules, readable factoring, and incremental testability.

## 3) Key architectural decisions

### 3.1 Control plane / daemon boundary

The primary architectural boundary is **daemon ↔ control plane**:

- The **control plane** owns authoritative state (event log + projections) and serves APIs to the desktop UI and `rn`.
- The **daemon** owns repo-local execution: git/worktrees, agent lifecycle, telemetry observation, and command execution.

Embedding both in the desktop app is an implementation detail:

- “Local embedded daemon” uses an **in-proc transport**.
- “Remote daemon” uses a **network transport**.
- The control plane speaks only to a **transport trait**, never daemon internals.

### 3.2 Serialization strategy (long-term view)

We optimize for performance while preserving debuggability:

- Protocol types are **strongly typed Rust structs**.
- In-proc transport can pass typed messages without serialization.
- Remote transports default to a **binary codec** (Protobuf) for perf.
- A **JSON codec** remains available as an opt-in dev/diagnostic mode, with tooling to decode/inspect traffic.

### 3.3 Identity strategy

Use **ULID + newtypes everywhere** in the Rust port.

- Wire format: ULID string (JSON) or 16 bytes (Protobuf).
- Storage format: prefer `BLOB(16)` for indices/perf; UI and logs render readable ULID strings.

### 3.4 Session viewer direction

We are moving toward a **native session viewer** that is a “wrapper around a session”, not a general terminal emulator.

- Shell/generic agents can continue to use tmux-oriented UX as a compatibility path.
- Exec- and app-server-style agents should be surfaced via a **scrollable session view + chat-like interaction**, with reusable components that can be shown:
  - in a left-hand “overseer agent” pane, and
  - on-demand from the graph (e.g., expanded task card / details).

### 3.5 Graph layout direction

We can replace ReactFlow/ELK with a Rust-native layout engine or our own deterministic layout algorithm.

Priorities:

- performance and simplicity,
- deterministic behavior,
- smooth relayout when task cards expand/collapse.

## 4) Domain map (top-level workstreams)

This epic will be executed as massively parallel work across these domains:

0) Foundations (Rust workspace + crate seams)
1) Protocol layer (typed messages, transports, codecs)
2) Control plane core (event log + projections + APIs)
3) Daemon/repo executor core (git/worktrees + leases + telemetry)
4) Agent runtime + session abstraction (tmux + exec + app-server)
5) Desktop UI shell (GPUI)
6) Graph view + layout engine
7) Session viewer + chat consolidation
8) Diff + change visualization subsystem
9) Integrations (Linear/GitHub)
10) Packaging + distribution (macOS/Linux)

We will discuss each domain and create detailed tickets before implementation.

## 5) Domain 0: Task map (Foundations)

- `epics/gpui/tasks/T-1/README.md`: Rust workspace bootstrap + crate boundaries.
- `epics/gpui/tasks/T-2/README.md`: ULID newtypes everywhere (`redesmyn_ids`).
- `epics/gpui/tasks/T-3/README.md`: Error/result conventions.
- `epics/gpui/tasks/T-4/README.md`: Logging/tracing foundations.
- `epics/gpui/tasks/T-5/README.md`: Typed config layer.
- `epics/gpui/tasks/T-6/README.md`: `sqlx` storage scaffolding.
- `epics/gpui/tasks/T-7/README.md`: Transport + codec scaffolding (Protobuf + JSON debug).
- `epics/gpui/tasks/T-8/README.md`: Rust `rn` skeleton (fast CLI harness).

Sequencing intent:

- Land T-1 first to establish workspace + crate seams.
- After T-1, execute the remaining Domain 0 tickets in parallel (one task per branch/worktree), coordinating only where explicitly noted in the “Dependencies / sequencing” sections.

## 6) Split-codebase strategy (during port)

While building `gpui`, we can keep the existing Python/TS implementation alongside the Rust workspace. The existing codebase serves as:

- a reference for behavior and product semantics,
- a source of truth for what currently exists,
- and a compatibility target when we choose “parity first” for a subdomain.

We will deliberately choose which parts to port, redesign, or drop as we proceed.

## 7) Domain 1: Task map (Protocol layer)

- `epics/gpui/tasks/T-9/README.md`: Protocol envelope + versioning + scopes.
- `epics/gpui/tasks/T-10/README.md`: Protobuf schemas + codegen pipeline.
- `epics/gpui/tasks/T-11/README.md`: Daemon ↔ control plane stream protocol (handshake, commands, telemetry, resync).
- `epics/gpui/tasks/T-12/README.md`: Client ↔ control plane API protocol over Unix socket (multiplexed requests + subscriptions).
- `epics/gpui/tasks/T-13/README.md`: Protocol tooling + wiretap (`rn protocol …`).
- `epics/gpui/tasks/T-14/README.md`: Artifact references + structured session events (contract).

Sequencing intent:

- Define the envelope and schema pipeline first (T-9, T-10).
- Then execute daemon stream protocol, client API protocol, and tooling in parallel (T-11..T-14).
