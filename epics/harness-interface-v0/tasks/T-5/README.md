# T-5 Conflict auto-assist + gated auto-resume (repo clean + agent turn complete)

## Metadata

```yaml
id: T-5
epic: harness-interface-v0
stacked_on: T-11
must_land_after:
- T-2
- T-4
branch:
  suggested: rn/harness-interface-v0/T-5-conflict-auto-assist
linear:
  issue_id: 97b1529a-24bb-4ddf-8856-67afbe5a4f9c
  identifier: RED-22
```

## Problem

When merge/restack hits a conflict, we currently require manual user intervention to:

- notice the conflict
- attach to the correct worktree
- read the error output
- resolve conflicts
- resume the merge run

We already have much of the information needed to help:

- merge run knows the blocked worktree and error
- the system can generate remediation guidance (attach command, what to do next)

But we lack one crucial capability:

- knowing when the agent is ready to receive the remediation message
- knowing when the agent has completed its turn, so resuming is safe and intentional

Historically we attempted to infer this from interactive tmux logs (prompt heuristics), but that is not reliable enough to safely gate automation.
This task is explicitly **structured-only**: automated conflict assist must depend on machine-readable turn boundaries.

## Goal

Enable an automated “conflict assist” loop when the selected agent kind supports it:

1) On merge/restack blocked by conflict:
   - send remediation message to the agent automatically (best-effort)
2) Before resuming:
   - require **repo validation**: conflict truly resolved / git state unblocked
   - require **agent validation**: agent has completed its turn (turn complete)
3) If both are satisfied:
   - automatically resume the merge run

If the selected agent kind cannot provide the needed capabilities, the behavior should remain manual (current UI).

## Using T-10 (structured exec + semantic events)

T-10 is the source of truth for “structured vs interactive” sessions and provides the signals this task must consume.

### Enabling structured mode (required)

Structured mode is **command-driven** (there is no separate interface-mode switch). Users must explicitly opt in by providing a harness command that enables machine-readable output:

- **Codex**: include `exec` and `--json`, e.g. `codex exec --json …`
- **Claude Code**: include both `--print` (or `-p`) and `--output-format stream-json`, e.g. `claude --print --output-format stream-json …`

If those flags are not present, the session is treated as interactive and this task must not attempt any automation.

### Signals available from T-10

When structured mode is enabled, T-10 ensures the AgentDriver can tail the structured stream and will persist/emit:

- `agent_session.agent_interface_mode == structured`
- `agent_session.external_session_ref` (Codex thread id / Claude session id when available)
- DB events: `agent.turn_started`, `agent.turn_completed`, `agent.assistant_message`
- `task.agent_session_update` snapshots reflecting the above

This task should gate auto-resume on these explicit events, not tmux prompt heuristics.

## Requirements

### 0) Structured-only support

Conflict assist automation is enabled only when the active agent session can provide structured semantic events (i.e. `agent_session.agent_interface_mode == structured`).
If the session is interactive / tmux-only (no structured stream), the system must fall back to the existing manual flow (no partial heuristics-based automation).

### 1) Gating rules (must enforce both)

Auto-resume may happen only if:

- repo executor validates the blocked worktree is unblocked and safe to continue, AND
- the agent semantic stream reports an explicit “turn complete” for the remediation turn (structured boundary events; not prompt heuristics)

If either is not satisfied:

- do not resume
- surface an actionable state (“waiting for agent”, “waiting for conflicts resolved”).

### 2) Delivery model

The remediation message delivery should:

- be queued and retried when the agent reports “ready for input” (derived from structured signals)
- use a transport that works for structured sessions (not tmux-specific send-keys; prefer resuming the external session id from `external_session_ref` where supported)
- after delivery, wait for the **next structured remediation turn** to complete before resuming:
  - prefer correlating by `external_turn_id` when provided, else use a conservative “observed after send time” boundary.
- include timeouts; on timeout, fall back to the manual flow with guidance.

### 3) UI behavior

In the task details (merge run callout):

- show whether conflict assist is active and what it is waiting on
- keep manual actions available (“Copy attach”, “Resume”) as fallback

### 4) Safety + idempotency

- This must not accidentally resume mid-edit.
- If the user resolves conflicts manually and resumes, the system should gracefully stop the “assist” loop.
- If the agent makes changes but does not resolve conflicts, repo validation must prevent auto-resume.
- Do not use “assume complete after N seconds” fallbacks; use explicit turn boundary events or time out to manual.

## Acceptance criteria

- For Codex/Claude Code (when enabled), conflicts trigger an automatic remediation message delivery.
- Auto-resume only triggers when both repo state is clean and the agent’s turn completed.
- The system never wedges indefinitely; it times out to a clear manual fallback state.
