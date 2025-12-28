# T-12 Harness adapter: Cursor

## Metadata

```yaml
id: T-12
stacked_on: T-15
node:
  branch: rn/agent-orchestration/T-12-harness-cursor
```

## Brief (local)

- Add a Cursor harness profile (launch/attach/capabilities) and verify it works end-to-end with the generic runner/session model.
- Define the best-effort attach story (tmux if applicable; otherwise “open in worktree” semantics).
- Document limitations where Cursor does not support hooks or terminal attach in a portable way.

## Acceptance Criteria

- Cursor is runnable via a profile (no bespoke code required unless validation proves necessary).
- `rn agent doctor cursor` passes (or reports explicit, actionable degraded-mode warnings).
- Limitations are explicit in UI/CLI output (no silent failure modes).
