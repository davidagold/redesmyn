# T-1 Harness interface + capabilities + GenericHarness

## Metadata

```yaml
id: T-1
epic: harness-interface-v0
stacked_on:
branch:
  suggested: rn/harness-interface-v0/T-1-harness-interface
```

## Problem

We have a working local-first agent runtime (tmux-backed sessions, attach metadata, prelude sending, etc.), but “harness integration” is still implicit:

- We can start a process and capture logs/output.
- We cannot reliably know **when the harness is ready for input** or when it has **completed its turn**.
- We cannot expose a stable contract for harness-specific improvements (conflict auto-assist, structured status, etc.) without baking assumptions into the generic runner.

We need an explicit **Harness interface** that:

- separates *terminal/session IO* from *harness semantics*
- can be implemented by harness-specific interface implementations (Codex, Claude Code)
- supports a “generic fallback” mode for arbitrary commands

## Goal

Define a minimal, strongly typed v0 harness interface, including:

- a semantic status model (agent turn + readiness)
- an explicit capability declaration
- a GenericHarness interface implementation that supports any command with only baseline capabilities

This interface should be used for publishing status updates and for issuing higher-level commands (e.g. “send message”) that are safe to offer in the UI.

## Research requirement

The implementer should do web research (docs + GitHub if open source) before finalizing the design.
The goal is to ground the v0 interface in what we can actually detect reliably, and to avoid coupling to brittle assumptions.

## Key design principles

### 1) Separate concerns: transport vs harness

- **Session transport** (tmux/pty/process):
  - starts/stops the process
  - sends keystrokes/text
  - captures output/logs
  - provides attach/log pointers
- **Harness interface implementation**:
  - interprets output/state into semantic events (“idle”, “thinking”, “turn complete”, “needs input”)
  - declares which advanced features are supported
  - provides higher-level operations when safe (e.g. “send message”, “interrupt”)

The harness implementation may *consume* session output and may request that text be sent, but it should not own the session transport.

### 2) Capabilities are explicit (runtime)

We need runtime introspection for both UI and server logic (especially across daemon ↔ control plane boundaries).
Capabilities should be a small, explicit data structure; do not rely on “method presence”.

### 3) GenericHarness exists so “any process” is still usable

Users must be able to run arbitrary commands.
If the harness kind is unknown (or user chooses Generic), we still provide:

- start/stop/restart
- attach
- log capture
- best-effort “send raw text” (if running inside an interactive session), *but* without semantics like “turn complete”.

Advanced features (like conflict auto-assist) must only be enabled when the selected harness advertises the required capabilities.

## Proposed v0 contract (shape, not final names)

### 1) Semantic states / events

We need a small vocabulary that is stable:

- `AgentTurnState` (example):
  - `unknown` (GenericHarness / insufficient signals)
  - `ready` (safe to send input)
  - `busy` (actively working)
  - `blocked` (waiting for user; or needs attention)
  - `completed` (agent asserts it is done for now)

Events should be emitted as structured messages (daemon → control plane → UI):

- “turn started”
- “turn completed”
- “ready for input” / “not ready”
- optionally: “notification received” (harness-specific)

### 2) Capability declaration

A minimal v0 `HarnessCapabilities` should include:

- `can_detect_ready_for_input`
- `can_detect_turn_complete`
- `can_send_text`
- `can_interrupt` (optional)
- `can_receive_notifications` (optional)

### 3) Implementation lifecycle

The harness implementation needs hooks to:

- initialize for a session
- consume new output (stream or polled)
- publish derived events

Keep this minimal; avoid an over-engineered plugin framework.

## Acceptance criteria

- There is a strongly typed harness interface and capability model usable from both local and daemon execution paths.
- GenericHarness implementation exists and supports arbitrary commands without assuming harness-specific semantics.
- Status published to the UI can represent “turn complete” vs “unknown” distinctly (even if Generic remains “unknown”).
- The interface is designed so Codex and Claude Code implementations can be added without changing callers.
