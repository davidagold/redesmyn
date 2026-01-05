# T-8 Refactor terminology + model: “Agent” is the program (not “Harness”)

## Metadata

```yaml
id: T-8
epic: harness-interface-v0
stacked_on:
branch:
  suggested: rn/harness-interface-v0/T-8-agent-terminology-refactor
```

## Problem

The epic is currently framed around “Harness interface v0”, but the term **harness** is overloaded and misleading in this codebase:

- In many contexts “harness” reads like a *test harness* or a *process runner* (tmux/pty/logging), not an *agent program* (Codex / Claude Code / generic CLI).
- The v0 abstraction we need is not “how to start a process”, but “how to interpret the agent program’s behavior” (ready/turn complete) and safely drive higher-level workflows (send message, interrupt, etc.).

Separately, the DB currently distinguishes `Agent` vs `AgentSession`, but in product terms the UI is oriented almost entirely around **sessions**:

- The dashboard consumes `EpicGraph.agentSessions` keyed by `taskId` (latest session per task).
- There are no API routes that expose `Agent` as a first-class resource; `AgentResponse` exists but is not used by the API.
- The runtime enforces agents as *per-task* identity (`display_name == "a-<task_id>"`), so the stable identity is effectively the task.

This makes “Agent” as a logical identity feel redundant, while “Harness” is doing semantic-duty we’d rather reserve for the program being driven.

## Goal

We will do the refactor (not just document it) so follow-on work can use a consistent mental model:

1) Treat the external program we launch and drive (Codex / Claude Code / generic command) as the **Agent**.
2) Avoid using “Harness” to describe semantic interpretation; reserve “transport/runtime” for tmux/pty/logging concerns.
3) Rename/deprecate the existing DB `Agent` construct (which is not product-facing) so “Agent” can refer to the program/interface layer.

This task is a prerequisite for revising T-1 to use “Agent” terminology and for implementing T-7 (AgentDriver) between T-1 and T-2.

## Proposed terminology (v0)

### Concepts

- **Agent**: the interactive CLI/tool being driven (Codex, Claude Code, generic command).
- **Agent session**: a single run instance of the agent for a task, with attach/log pointers and runtime status.
- **Transport**: tmux/pty/process IO; knows how to send text/interrupt and collect output/logs.
- **Interpreter**: consumes output and produces semantic state (`turn_state`, readiness) + capability declaration.
- **Driver**: owns an interpreter for a session, consumes incremental output, persists/broadcasts semantic status, and issues safe high-level operations via transport when supported.

### Naming guidance (code)

We should avoid naming the semantic abstraction `Harness` if it causes confusion with transport/runtime. Prefer names that communicate intent:

- `Agent` / `AgentInterpreter` / `AgentSemantics` (instead of `Harness`)
- `AgentTransport` (instead of `HarnessTransport`)
- `AgentDriver` / `AgentMonitor` for the long-running loop (T-7)

Note: the existing DB `Agent` model name is already taken today. This task makes room by renaming it to a non-product-facing concept (e.g. `TaskAgent` / `AgentIdentity`) and updating call sites.

## Requirements

### 1) Document the terminology shift (glossary)

- Update the epic control doc (`epics/harness-interface-v0/README.md`) to include a short glossary aligning “Agent / session / transport / interpreter / driver”.
- Record the rationale succinctly: “UI and workflows are session-first; DB Agent is legacy/internal; ‘Harness’ is overloaded.”

### 2) Refactor DB `Agent` naming/role (free up “Agent”)

Implement the refactor (minimum viable) so code can use “Agent” to mean the program/interface layer:

- Rename the SQLAlchemy ORM `Agent` class to `TaskAgent` (or `AgentIdentity`) and update imports/usages.
- Keep the underlying DB table name (`agents`) for now unless the migration is trivial and low-risk.
- Audit and update join points that only exist to read `agent.display_name` / `agent_id` for warnings:
  - merge planning “running agents” warnings (currently `AgentSession ⨝ Agent`) should remain possible but use the renamed model.
- Ensure no public API surface is framed around the legacy identity object:
  - `AgentResponse` exists but is not used by the API today; remove it or rename it to match the new internal naming.
- Review CLI surfaces that currently use the legacy identity name:
  - `rn agent register` / `rn agent list` are identity-oriented; rename/move them under a more explicit internal namespace (e.g. `rn task-agent …`) or remove if unused.
  - Keep task workflows (`rn agent start|restart|stop`, dashboard controls) framed around sessions and “Agent (program)” semantics.

This refactor should explicitly unblock:

- T-1 revision: “Agent interface + capabilities + generic fallback” without semantic collision.
- T-7: “AgentDriver” can talk about driving “the Agent” (the program), not a “Harness”.
- T-6: doctor/capabilities surfaces can describe “Agent kind/capabilities” clearly.

### 3) Align downstream task framing and copy

- Revise T-1 (in its own branch) to use the clarified terms (Agent interface/interpreter + capabilities + generic fallback) and keep “transport vs semantics” separation crisp.
- Keep “harness profile” terminology only where it truly means “how to launch” (argv/env/cwd); avoid using it to mean semantic behavior.
- Ensure CLI/UI copy prefers “Agent” for the program and “Session” for a run instance.

### Non-goals (for T-8)

- Do not implement AgentDriver (that is T-7).
- Do not implement Codex/Claude detection (T-3/T-4).
- Do not remove the `agents` table entirely unless it is strictly necessary to avoid collisions; that can be a later cleanup once sessions/configs are fully task-scoped.

## Acceptance criteria

- The epic control doc includes a crisp glossary and rationale for the terminology/model alignment.
- The codebase no longer uses the name `Agent` for the legacy DB identity object (renamed to `TaskAgent`/`AgentIdentity` or removed), and call sites are updated.
- The task map/order is preserved: T-8 → revised T-1 → T-7 (AgentDriver) → T-2/T-3/T-4.
