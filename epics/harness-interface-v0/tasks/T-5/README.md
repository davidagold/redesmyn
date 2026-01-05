# T-5 Conflict auto-assist + gated auto-resume (repo clean + agent turn complete)

## Metadata

```yaml
id: T-5
epic: harness-interface-v0
stacked_on: T-3
branch:
  suggested: rn/harness-interface-v0/T-5-conflict-auto-assist
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

- knowing when the harness is ready to receive the remediation message
- knowing when the agent has completed its turn, so resuming is safe and intentional

## Goal

Enable an automated “conflict assist” loop when the selected harness supports it:

1) On merge/restack blocked by conflict:
   - send remediation message to the agent automatically (best-effort)
2) Before resuming:
   - require **repo validation**: conflict truly resolved / git state unblocked
   - require **harness validation**: agent has completed its turn (turn complete)
3) If both are satisfied:
   - automatically resume the merge run

If the harness cannot provide the needed capabilities, the behavior should remain manual (current UI).

## Requirements

### 1) Gating rules (must enforce both)

Auto-resume may happen only if:

- repo executor validates the blocked worktree is unblocked and safe to continue, AND
- harness adapter reports “turn complete” for the remediation turn

If either is not satisfied:

- do not resume
- surface an actionable state (“waiting for agent”, “waiting for conflicts resolved”, or “unknown”)

### 2) Delivery model

The remediation message delivery should:

- be queued and retried when the harness reports “ready for input”
- include a timeout; if delivery cannot be confirmed, fall back to manual flow with guidance

### 3) UI behavior

In the task details (merge run callout):

- show whether conflict assist is active and what it is waiting on
- keep manual actions available (“Copy attach”, “Resume”) as fallback

### 4) Safety + idempotency

- This must not accidentally resume mid-edit.
- If the user resolves conflicts manually and resumes, the system should gracefully stop the “assist” loop.
- If the agent makes changes but does not resolve conflicts, repo validation must prevent auto-resume.

## Acceptance criteria

- For Codex/Claude Code (when enabled), conflicts trigger an automatic remediation message delivery.
- Auto-resume only triggers when both repo state is clean and the agent’s turn completed.
- The system never wedges indefinitely; it times out to a clear manual fallback state.

