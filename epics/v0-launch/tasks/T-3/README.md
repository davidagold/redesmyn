# T-3 Remove local mode: make the control plane pure

## Metadata

```yaml
id: T-3
stacked_on: T-2
node:
  branch: rn/v0-launch/T-3-remove-local-mode
```

## Brief (local)

Remove the hybrid “local mode” where the server can execute repo-local work:

- Delete or deprecate `runner_mode=local` and the corresponding in-process behaviors:
  - server-owned repo observer loop
  - server-owned agent monitor/driver loop
  - server-owned primary executor lease refresh
- Make repo-local execution always route to an attached daemon (even on localhost).
- Ensure the server can run without repo filesystem access.

## Acceptance Criteria

- The server process starts with no repo FS access (only DB access) and remains functional for read-only UI/API.
- All endpoints that require repo-local execution fail with actionable guidance when no daemon is connected/attached.
- No background tasks in the server process perform git/worktree inspection or agent supervision.

