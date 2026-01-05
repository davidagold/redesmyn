# T-0 Terminology + model alignment: “Agent program” (not “Harness”)

## Metadata

```yaml
id: T-0
epic: harness-interface-v0
stacked_on:
branch:
  suggested: rn/harness-interface-v0/T-0-agent-terminology-base
```

## Problem

The epic is currently framed around “Harness interface v0”, but the term **harness** is overloaded and misleading in this codebase:

- In many contexts “harness” reads like a *test harness* or a *process runner* (tmux/pty/logging), not an *agent program* (Codex / Claude Code / generic CLI).
- The v0 abstraction we need is not “how to start a process”, but “how to interpret the agent program’s behavior” (ready/turn complete) and safely drive higher-level workflows (send message, interrupt, etc.).

Separately, the DB currently distinguishes `Agent` vs `AgentSession`, but in product terms the UI is oriented almost entirely around **sessions**:

- The dashboard consumes `EpicGraph.agentSessions` keyed by `taskId` (latest session per task).
- There are no API routes that expose `Agent` as a first-class resource; `AgentResponse` exists but is not used by the API.
- The runtime enforces agents as *per-task* identity (`display_name == "a-<task_id>"`), so the stable identity is effectively the task.

This makes “Agent” as a logical identity feel redundant, while “Harness” is doing semantic-duty we’d rather reserve for “the program being driven”.

## Goal

Make the epic’s language and model intent explicit so follow-on work can be consistent:

1) Treat the external program we launch and drive (Codex / Claude Code / generic command) as the **Agent program** in the conceptual model.
2) Avoid using “Harness” to describe semantic interpretation; reserve “transport/runtime” for tmux/pty/logging concerns.
3) Create space to represent the Agent program semantics (and its driver) without colliding with the existing DB `Agent` identity concept.

This task is a prerequisite for revising T-1 to use the clarified terminology and for adding T-7 (AgentDriver) between T-1 and T-2.

## Proposed terminology (v0)

### Concepts

- **Agent program**: the interactive CLI/tool being driven (Codex, Claude Code, generic command).
- **Agent session**: a single run instance of the agent program for a task, with attach/log pointers and runtime status.
- **Transport**: tmux/pty/process IO; knows how to send text/interrupt and collect output/logs.
- **Interpreter**: consumes output and produces semantic state (`turn_state`, readiness) + capability declaration.
- **Driver**: owns an interpreter for a session, consumes incremental output, persists/broadcasts semantic status, and issues safe high-level operations via transport when supported.

### Naming guidance (code)

We should avoid naming a semantic abstraction `Harness` if it causes confusion with transport/runtime. Prefer names that communicate intent:

- `AgentInterpreter` / `AgentSemantics` (instead of `Harness`)
- `AgentTransport` (instead of `HarnessTransport`)
- `AgentDriver` / `AgentMonitor` for the long-running loop (T-7)

Note: the existing DB `Agent` model name is already taken; we can either keep it as legacy/per-task identity or plan a follow-up migration to remove it. This task does not require the migration, but it should make that option explicit.

## Requirements

### 1) Document the terminology shift

- Update the epic control doc (`epics/harness-interface-v0/README.md`) to include a short glossary aligning “Agent program / session / transport / interpreter / driver”.
- Record the rationale: “Agent (DB identity) is not surfaced; UI and workflows are session-first; ‘Harness’ is overloaded.”

### 2) Decide how to handle DB `Agent` going forward (decision record)

Capture one of:

- **Option A (minimal change, near-term)**: keep DB `Agent` as internal/per-task identity + pointer to `current_session_id`, but do not reuse “Agent” to describe the external program; use “Agent program” for that.
- **Option B (simplify model, longer-term)**: plan to migrate toward task-scoped sessions/configs and remove/fully deprecate DB `Agent` as a separate table once legacy constraints are resolved.

This decision should explicitly call out which follow-on tasks depend on it (T-1 revision, T-7 driver, T-6 doctor surfaces).

### 3) Align downstream task framing

- Revise T-1 (in its own PR/branch) to use the clarified terms (interpreter + capabilities + generic fallback), and to keep “transport vs semantics” separation crisp.
- Add T-7 (in its own PR/branch) to implement the driver/monitor loop between T-1 and T-2.

## Acceptance criteria

- The epic control doc includes a crisp glossary and rationale for the terminology/model alignment.
- There is an explicit recorded decision about the DB `Agent` table’s role (keep for now vs migrate away), with stated implications.
- The task map reflects the intended ordering: Base (T-0) → revised T-1 → T-7 (AgentDriver) → T-2/T-3/T-4.

