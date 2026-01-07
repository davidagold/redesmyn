# T-8 Remove DB `Agent` construct; “Agent” is the program (not “Harness”)

## Metadata

```yaml
id: T-8
epic: harness-interface-v0
stacked_on: null
branch:
  suggested: rn/harness-interface-v0/T-8-remove-agent-construct
linear:
  issue_id: 262fc098-360b-4047-a2e6-c66ff26f07e0
  identifier: RED-25
```

## Problem

The epic is currently framed around “Harness interface v0”, but the term **harness** is overloaded and misleading in this codebase:

- In many contexts “harness” reads like a *test harness* or a *process runner* (tmux/pty/logging), not an *agent program* (Codex / Claude Code / generic CLI).
- The v0 abstraction we need is not “how to start a process”, but “how to interpret the agent program’s behavior” (ready/turn complete) and safely drive higher-level workflows (send message, interrupt, etc.).

Separately, the DB currently distinguishes `Agent` vs `AgentSession`, but in product terms the UI is oriented almost entirely around **sessions**:

- The dashboard consumes `EpicGraph.agentSessions` keyed by `taskId` (latest session per task).
- There are no API routes that expose `Agent` as a first-class resource; `AgentResponse` exists but is not used by the API.
- The runtime enforces agents as *per-task* identity (`display_name == "a-<task_id>"`), so the stable identity is effectively the task.

This makes the DB `Agent` identity construct redundant, while “Harness” is doing semantic-duty we’d rather reserve for the program being driven.

## Goal

Remove the DB `Agent` construct altogether so follow-on work can use a consistent, product-aligned model:

1) Treat the external program we launch and drive (Codex / Claude Code / generic command) as the **Agent**.
2) Treat **AgentSession** as the canonical runtime entity for “agent state”, keyed/anchored by `task_id`.
3) Remove the legacy DB `Agent` table and references (`tasks.agent_id`, `agent_configs.agent_id`, `agent_sessions.agent_id`, `current_session_id` pointer) by moving the remaining useful state to task/session-scoped structures.
4) Avoid using “Harness” to describe semantic interpretation; reserve “transport/runtime” for tmux/pty/logging concerns.

This task is a prerequisite for revising T-1 to use “Agent” terminology and for implementing T-7 (AgentDriver) between T-1 and T-2.

## Proposed terminology (v0)

### Concepts

- **Agent**: the interactive CLI/tool being driven (Codex, Claude Code, generic command).
- **Agent session**: a single run instance of the agent for a task, with attach/log pointers and runtime status.
- **Transport**: tmux/pty/process IO; knows how to send text/interrupt and collect output/logs.
- **Interpreter**: consumes output and produces semantic state (`turn_state`, readiness) + capability declaration.
- **Driver**: owns an interpreter for a session, consumes incremental output, persists/broadcasts semantic status, and issues safe high-level operations via transport when supported.

### Model guidance (code)

We should avoid naming the semantic abstraction `Harness` if it causes confusion with transport/runtime. Prefer names that communicate intent:

- `Agent` / `AgentInterpreter` / `AgentSemantics` (instead of `Harness`)
- `AgentTransport` (instead of `HarnessTransport`)
- `AgentDriver` / `AgentMonitor` for the long-running loop (T-7)

The key requirement is removal of the legacy DB identity object entirely; we should not rename-and-keep it unless we discover a compelling product need for a cross-session identity (we currently have none).

## Requirements

### 1) Update docs/glossary (terminology)

- Update the epic control doc (`epics/harness-interface-v0/README.md`) to include a short glossary aligning “Agent / session / transport / interpreter / driver”.
- Record the rationale succinctly: “UI and workflows are session-first; DB Agent is legacy/internal; ‘Harness’ is overloaded.”

### 2) Remove DB `Agent` table and references

Implement the structural refactor so `AgentSession` and `Task` fully cover what the product needs.

Concrete changes (minimum set to delete the construct):

- **Make `AgentSession` task-anchored and self-contained**
  - Remove `agent_sessions.agent_id` (and any join dependency on `agents`).
  - Ensure all session-facing UI/API fields can be derived from `task_id` + session state.
  - If a “display name” is needed for logs/warnings, derive it deterministically from `task_id` (e.g. `a-<task_id>`) or store a denormalized `agent_label` directly on the session.

- **Make `AgentConfig` task-scoped or session-scoped**
  - Replace `agent_configs.agent_id` with `agent_configs.task_id` (unique) OR eliminate the table and persist config snapshots only on `AgentSession` (preferred if feasible).
  - Ensure restart uses a predictable source of truth:
    - either the task-scoped config row, or
    - the latest session’s `resolved_launch_configuration`/config snapshot.

- **Remove `tasks.agent_id`**
  - Replace “task has agent?” checks with “task has a (running) latest session?” and/or “task has config?” depending on workflow.
  - Update any “task.agent_set” style eventing if it exists; the concept should become session-driven.

- **Remove `current_session_id`**
  - Replace `agents.current_session_id` with either:
    - `tasks.current_agent_session_id` (if you truly need a pointer), or
    - derive “current” as “latest non-ended session” by query (preferred unless performance becomes an issue).

- **Update filesystem layout and paths**
  - Adjust log/session directory paths to not require an `agent_id` prefix.
  - Replace `.../.redesmyn/agents/<agent_id>/sessions/<session_id>/output.log` with a task/session anchored path, e.g.:
    - `.../.redesmyn/tasks/<task_id>/agent-sessions/<session_id>/output.log`, or
    - `.../.redesmyn/agent-sessions/<session_id>/output.log` (if `session_id` is already unique).
  - Ensure attach info continues to point at the correct log path.

- **Remove/replace CLI commands that create/list agents**
  - `rn agent register` / `rn agent list` are identity-oriented and should be removed or re-homed under a clearly internal namespace if still useful.
  - Keep task workflows (`rn agent start|restart|stop`, dashboard controls) framed around sessions and “Agent (program)” semantics.

- **Data migration and backwards compatibility**
  - Provide a forward migration from existing DBs:
    - translate `tasks.agent_id` + `agents` rows into session/config/task-scoped state,
    - ensure existing sessions remain queryable and attach/log UX continues to work.
  - Update the legacy upgrade/backfill logic in `redesmyn/db/session.py` to match the new schema and remove “backfill from agents table” once the table is gone.

### 3) Replace “Agent” joins with session/task-derived fields

Audit and update join points that exist solely for:

- retrieving `agent.display_name` (warnings, UI labels),
- using `agent.current_session_id` as a pointer.

Examples to update:

- Epic graph response currently joins `AgentSession ⨝ Agent` to provide `agent_name`.
- Merge planning “running agents” detection joins to read `agent_name` for warnings.

After this refactor, these should be derived from `task_id` (or session-local fields) with no join.

This refactor should explicitly unblock:

- T-1 revision: “Agent interface + capabilities + generic fallback” without semantic collision.
- T-7: “AgentDriver” can talk about driving “the Agent” (the program), not a “Harness”.
- T-6: doctor/capabilities surfaces can describe “Agent kind/capabilities” clearly.

### 3) Align downstream task framing and copy

- Revise T-1 (in its own branch) to use the clarified terms (Agent interface/interpreter + capabilities + generic fallback) and keep “transport vs semantics” separation crisp.
- Keep “launch configuration” terminology only where it truly means “how to launch” (argv/env/cwd); avoid using it to mean semantic behavior.
- Ensure CLI/UI copy prefers “Agent” for the program and “Session” for a run instance.

### Non-goals (for T-8)

- Do not implement AgentDriver (that is T-7).
- Do not implement Codex/Claude detection (T-3/T-4).
- Do not add new “identity” constructs unless a concrete product requirement emerges (multi-user scheduling, shared fleets, etc.).

## Acceptance criteria

- The epic control doc includes a crisp glossary and rationale for the terminology/model alignment.
- The `agents` table is removed from the schema (and corresponding ORM/API/CLI constructs are removed).
- `AgentSession` and task-scoped/session-scoped config fully cover start/restart/stop/attach/logs flows.
- The dashboard/CLI still show “latest session per task” correctly and running-agent safety checks still work without joining to an `agents` table.
- The task map/order is preserved: T-8 → revised T-1 → T-7 (AgentDriver) → T-2/T-3/T-4.
