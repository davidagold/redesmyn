# T-1 Agent interface + capabilities + ShellAgent

## Metadata

```yaml
id: T-1
epic: harness-interface-v0
stacked_on: T-8
branch:
  suggested: rn/harness-interface-v0/T-1-agent-interface
linear:
  issue_id: fbba86a5-c122-4ac4-9494-797523a724f2
  identifier: RED-17
```

## Problem

We have a working local-first session runtime (tmux-backed sessions, attach metadata, prelude sending, etc.), but “agent program integration” is still implicit:

- We can start a process and capture logs/output.
- We cannot reliably know **when the agent is ready for input** or when it has **completed its turn**.
- We cannot expose a stable contract for agent-specific improvements (conflict auto-assist, structured status, etc.) without baking assumptions into the generic runner.

We need an explicit **Agent interface** (semantics/interpreter) that:

- separates *terminal/session IO* from *agent semantics*
- can be implemented by agent-specific interface implementations (Codex, Claude Code)
- supports a “generic fallback” mode for arbitrary commands

## Goal

Define a minimal, strongly typed v0 agent interface, including:

- a semantic status model (agent turn + readiness)
- an explicit capability declaration
- a ShellAgent interface implementation that supports any command with only baseline capabilities

This interface should be used for publishing status updates and for issuing higher-level commands (e.g. “send message”) that are safe to offer in the UI.

## Research requirement

The implementer should do web research (docs + GitHub if open source) before finalizing the design.
The goal is to ground the v0 interface in what we can actually detect reliably, and to avoid coupling to brittle assumptions.

## Key design principles

### 1) Separate concerns: transport vs agent semantics

- **Session transport** (tmux/pty/process):
  - starts/stops the process
  - sends keystrokes/text
  - captures output/logs
  - provides attach/log pointers
- **Agent interface implementation** (interpreter):
  - interprets output/state into semantic events (“idle”, “thinking”, “turn complete”, “needs input”)
  - declares which advanced features are supported
  - provides higher-level operations when safe (e.g. “send message”, “interrupt”)

The agent implementation may *consume* session output and may request that text be sent, but it should not own the session transport.

### 2) Capabilities are explicit (runtime)

We need runtime introspection for both UI and server logic (especially across daemon ↔ control plane boundaries).
Capabilities should be a small, explicit data structure; do not rely on “method presence”.

### 3) ShellAgent exists so “any process” is still usable

Users must be able to run arbitrary commands.
If the agent kind is unknown (or user chooses Generic), we still provide:

- start/stop/restart
- attach
- log capture
- best-effort “send raw text” (if running inside an interactive session), *but* without semantics like “turn complete”.

Advanced features (like conflict auto-assist) must only be enabled when the selected agent advertises the required capabilities.

## Proposed v0 contract (shape, not final names)

### 1) Semantic states / events

We need a small vocabulary that is stable:

- `AgentTurnState` (example):
  - `unknown` (ShellAgent / insufficient signals)
  - `ready` (safe to send input)
  - `busy` (actively working)
  - `blocked` (waiting for user; or needs attention)
  - `completed` (agent asserts it is done for now)

Events should be emitted as structured messages (daemon → control plane → UI):

- “turn started”
- “turn completed”
- “ready for input” / “not ready”
- optionally: “notification received” (agent-specific)

### 1.1) External session handles (resume)

We need a place to store the agent program’s own “session identifier” when available.
This is distinct from Redesmyn’s `AgentSession.id` (DB primary key) and is used to resume a conversation in a new process.

Key findings (Jan 2026):

- **Codex**: `thread_id` is the session identifier; `turn_id` is a per-turn identifier.
- **Claude Code**: JSON output includes a `session_id`, and the CLI supports resuming by id (and/or “continue in cwd”).

The v0 interface should provide a typed way to surface this (examples, not final names):

- `AgentExternalSessionRef`:
  - `kind` (`codex_thread` | `claude_session` | `unknown`)
  - `id` (string)
  - optional `turn_id` (string; informational only)

This should be persisted on `AgentSession` (or adjacent session-scoped state) so the UI/CLI can offer “resume” affordances when supported.

### 2) Capability declaration

A minimal v0 `AgentCapabilities` should include:

- `can_detect_ready_for_input`
- `can_detect_turn_complete`
- `can_send_text`
- `can_interrupt` (optional)
- `can_receive_notifications` (optional)
- `can_resume_by_id` (optional; depends on agent kind)
- `can_continue_in_cwd` (optional; depends on agent kind)

### 3) Implementation lifecycle

The agent implementation needs hooks to:

- initialize for a session
- consume new output (stream or polled)
- publish derived events

Keep this minimal; avoid an over-engineered plugin framework.

## Acceptance criteria

- There is a strongly typed agent interface and capability model usable from both local and daemon execution paths.
- ShellAgent implementation exists and supports arbitrary commands without assuming agent-specific semantics.
- Status published to the UI can represent “turn complete” vs “unknown” distinctly (even if Generic remains “unknown”).
- The interface is designed so Codex and Claude Code implementations can be added without changing callers.
