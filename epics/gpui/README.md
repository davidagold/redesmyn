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

Agent runtime priority (implementation order for this epic):

1) **Codex (structured)** (highest priority)
2) **Shell** (tmux; interactive/unstructured)
3) **Claude Code** (structured)
4) **App-server** agents (future-facing; skeleton in this epic)

Session persistence rule:

- Persist all **non-delta / non-chunk** session emissions needed to render conversation history and actionable structured timelines in the native session viewer.

Desktop shell layout direction (GPUI):

- Remove the always-visible left navigation sidebar from the web UI.
- Use the saved width budget for a **persistently visible, collapsible left pane** that hosts a session view for a user-managed chat pinned to the selected epic (“overseer” in developer shorthand, but **not** a user-facing construct).
- Pin semantics: an epic has **0/1** pinned chat session; a chat session can be pinned to **0..N** epics.
- The right pane remains graph-first (graph + details).
- Intentionally omit “target the selected task by default” coupling from the port; leave it as a future UX improvement.

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
- `epics/gpui/tasks/T-15/README.md`: AI-first testability surfaces (wait primitives + semantic UI snapshot + UI driver contract).

Sequencing intent:

- Define the envelope and schema pipeline first (T-9, T-10).
- Then execute daemon stream protocol, client API protocol, and tooling in parallel (T-11..T-14).
- Treat testability as a first-class requirement: T-15 defines the automation surfaces early so later domains can build against stable contracts.

## 8) Domain 2: Task map (Control plane core)

- `epics/gpui/tasks/T-16/README.md`: Headless control plane service skeleton.
- `epics/gpui/tasks/T-17/README.md`: Control plane DB schema + migrations (sqlx).
- `epics/gpui/tasks/T-18/README.md`: Event log append + subscription hub.
- `epics/gpui/tasks/T-19/README.md`: Command engine (persisted lifecycle + routing to daemon).
- `epics/gpui/tasks/T-20/README.md`: Client API server over UDS (requests + subscriptions).
- `epics/gpui/tasks/T-21/README.md`: Epic graph query model + projection.
- `epics/gpui/tasks/T-22/README.md`: Control plane integration test harness (mock daemon + real repo modes).

Sequencing intent:

- Establish schema + minimal service skeleton early (T-16, T-17).
- Build the event pipeline and command engine next (T-18, T-19).
- Expose the client API over UDS and validate with integration tests (T-20, T-22).
- Implement and tune the EpicGraph read model as the primary UI/CLI query (T-21).

## 9) Domain 3: Task map (Daemon / repo executor core)

- `epics/gpui/tasks/T-23/README.md`: Daemon runtime skeleton (service + control-plane connection).
- `epics/gpui/tasks/T-24/README.md`: Repo registry + attachment semantics (no path leakage).
- `epics/gpui/tasks/T-25/README.md`: Lease/primary executor management + enforcement.
- `epics/gpui/tasks/T-26/README.md`: Git backend abstraction (CLI-first now, swappable later).
- `epics/gpui/tasks/T-27/README.md`: Worktree management service.
- `epics/gpui/tasks/T-28/README.md`: Repo observation + telemetry + snapshots.
- `epics/gpui/tasks/T-29/README.md`: Merge/restack planner (deterministic plans).
- `epics/gpui/tasks/T-30/README.md`: Merge/restack executor (resumable, step updates).
- `epics/gpui/tasks/T-31/README.md`: Daemon integration test harness (real repo fixtures).

Sequencing intent:

- Stand up the daemon runtime first (T-23), then enable safe attachment + identity (T-24) and lease enforcement (T-25).
- Implement the git/worktree substrate (T-26, T-27), then observation/telemetry (T-28).
- Build merge/restack as a plan+execute split (T-29, T-30) so UI/CLI can preview plans and execution can be resumable.
- Keep real-repo integration tests close to the daemon implementation (T-31) to enforce AI-friendly determinism.

## 10) Domain 4: Task map (Agent runtime + sessions)

This domain ports/redesigns agent execution around **durable structured session events** (T-14) while preserving the key behaviors we rely on today:

- resume-by-id turns (structured agents),
- “send message” conflict handling semantics,
- interrupt semantics.

Tasks:

- `epics/gpui/tasks/T-32/README.md`: Agent kind taxonomy + interface mode inference + resume-by-id turn builder (Shell naming).
- `epics/gpui/tasks/T-33/README.md`: Codex structured parser → session events (pure state machine + tests).
- `epics/gpui/tasks/T-34/README.md`: Claude Code structured parser → session events (pure state machine + tests).
- `epics/gpui/tasks/T-35/README.md`: Daemon exec-session supervisor (process lifecycle + event streaming + backpressure).
- `epics/gpui/tasks/T-36/README.md`: Daemon Shell (tmux) session runtime (attach/send/interrupt + log artifacts).
- `epics/gpui/tasks/T-37/README.md`: Daemon Codex runner (structured) (exec-based; output-last-message capture).
- `epics/gpui/tasks/T-38/README.md`: Daemon Claude Code runner (structured) (exec-based).
- `epics/gpui/tasks/T-39/README.md`: Daemon app-server agent runtime skeleton.
- `epics/gpui/tasks/T-40/README.md`: Control plane session persistence + query surfaces (sqlx).
- `epics/gpui/tasks/T-41/README.md`: Control plane agent commands + “send message” semantics (conflicts/resume/interrupt) + API methods.
- `epics/gpui/tasks/T-42/README.md`: End-to-end agent session integration tests (mock agents, determinism, persistence).

Sequencing intent:

- Build parser + argv utilities early (T-32..T-34) so runtime implementers have stable building blocks.
- Stand up daemon execution substrate (T-35) before agent-specific runners (T-37/T-38).
- Implement Shell/tmux support as a separate, compatibility-focused track (T-36).
- Land control-plane persistence + agent command semantics early enough that UI/CLI work can proceed without inventing ad-hoc plumbing (T-40/T-41).

## 11) Domain 5: Task map (Desktop UI shell — GPUI)

This domain replaces the web dashboard with a native GPUI desktop app.

Layout direction:

- No persistent navigation sidebar.
- A persistently visible, collapsible **left session pane** scoped to the selected epic.
- A graph-first right pane.
- “Overseer” is developer shorthand only; the UI does not introduce a new user-facing construct.

Tasks:

- `epics/gpui/tasks/T-43/README.md`: GPUI desktop app bootstrap + lifecycle (embed control plane + daemon modules).
- `epics/gpui/tasks/T-44/README.md`: GPUI UI foundations (theme, tokens, gpui-component survey, shared widgets).
- `epics/gpui/tasks/T-45/README.md`: Main split layout (left session pane + right workspace; resizable + collapsible).
- `epics/gpui/tasks/T-46/README.md`: Epic header + chrome (epic selector, status, refresh, settings/command entrypoints).
- `epics/gpui/tasks/T-47/README.md`: User-managed chat sessions + epic pins (no “overseer” naming).
- `epics/gpui/tasks/T-48/README.md`: Desktop UI driver + semantic UI snapshot (AI-first testability; local-only).
- `epics/gpui/tasks/T-49/README.md`: Command palette skeleton (upgrade; not required for initial port).

Sequencing intent:

- Stand up the GPUI app host first (T-43).
- Build layout + chrome in parallel (T-44..T-46).
- Define the epic-scoped session selection and persistence seam early so Session Viewer work can plug in cleanly later (T-47).
- Implement test driver hooks early enough that subsequent UI work is easy for AI/CI to validate (T-48).

## 12) Domain 6: Task map (Graph view + layout engine)

This domain ports the graph-first UI to GPUI and replaces ReactFlow/ELK with a Rust-native renderer and layout engine.

Core requirements:

- Deterministic layout (stable across runs for the same input).
- Smooth, responsive interactions (pan/zoom/select) with clear selection state.
- Support dynamic relayout when task cards expand/collapse (variable node sizes).
- “No silent actions” (graph-triggered mutations must show immediate in-flight feedback).
- AI-first testability: the graph must be driveable and assertable via semantic UI snapshot surfaces (T-48).

Tasks:

- `epics/gpui/tasks/T-50/README.md`: Graph scene + renderer scaffolding (GPUI canvas, camera, hit-testing).
- `epics/gpui/tasks/T-51/README.md`: Deterministic layout engine v1 (variable node sizes; expand/collapse relayout).
- `epics/gpui/tasks/T-52/README.md`: Task node view (compact/expanded; measurement; selection affordances).
- `epics/gpui/tasks/T-53/README.md`: Edge routing + rendering (orthogonal edges; hover/selection; LOD labels).
- `epics/gpui/tasks/T-54/README.md`: Viewport behaviors (fit-to-view, pan-to-selection, focus mode path).
- `epics/gpui/tasks/T-55/README.md`: Details panel (drawer) + selection model integration.
- `epics/gpui/tasks/T-56/README.md`: Bulk selection + action bar (multi-select UX).
- `epics/gpui/tasks/T-57/README.md`: Trunk timeline column (commit marks; base alignment; optional but planned).
- `epics/gpui/tasks/T-58/README.md`: Graph testability surfaces (extend UI driver + semantic snapshot for graph).

Sequencing intent:

- Land renderer + layout foundations early (T-50, T-51).
- Build node/edge rendering and viewport behaviors in parallel (T-52..T-54).
- Integrate details and bulk actions after selection/interaction are stable (T-55, T-56).
- Keep trunk timeline optional so it doesn’t block core graph parity (T-57).

## 13) Domain 7: Task map (Session viewer + chat consolidation)

This domain builds the reusable native **session viewer** used for:

- user-managed chat sessions pinned to the selected epic (left pane), and
- task-scoped agent sessions in the graph/details surface.

Core requirements:

- Full markdown rendering for messages.
- Session semantics: **session == conversation; turns are events** (event-as-turn).
- Scroll performance: virtualized feed + stable scroll anchoring.
- “No silent actions”: visible in-flight state for send/load/attach flows.
- AI-first testability: driveable via UI driver + assertable via semantic snapshots (T-48).

Tasks:

- `epics/gpui/tasks/T-59/README.md`: Session viewer foundations (event→view model + pagination + subscriptions).
- `epics/gpui/tasks/T-60/README.md`: Full markdown rendering for session messages (GPUI).
- `epics/gpui/tasks/T-61/README.md`: SessionView virtualized feed + scroll behaviors.
- `epics/gpui/tasks/T-62/README.md`: Session composer + conflict/confirm UX (preserve v0 semantics).
- `epics/gpui/tasks/T-63/README.md`: Left pane pinned chat session viewer (no “overseer” naming).
- `epics/gpui/tasks/T-64/README.md`: Task details session view (latest session only).
- `epics/gpui/tasks/T-65/README.md`: Interactive (tmux) session placeholder UX (attach/copy; no terminal emulator in port).
- `epics/gpui/tasks/T-66/README.md`: Session viewer AI-testability (UI driver actions + semantic snapshot + tests).
