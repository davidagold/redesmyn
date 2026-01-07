# T-6 `rn daemon up/down/status`: background daemon + attach/registry ergonomics

## Metadata

```yaml
id: T-6
stacked_on: T-5
node:
  branch: rn/v0-launch/T-6-daemon-up-down-status
```

## Brief (local)

Implement the v0 daemon lifecycle UX described in revise-architecture T-4:

- `rn daemon up`: start (or ensure) a background daemon process for this host.
- `rn daemon down`: stop it.
- `rn daemon status`: show whether it’s online, where it’s connected, and what repos are attached.

Also implement “repo attachment” ergonomics:

- Automatic repo registration/lookup for `workspace_id + repo_id` (stored host-locally).
- `rn` commands that require execution should ensure attach (or emit actionable guidance).
- Provide explicit “claim/take primary” affordances if another host holds the lease (v0 can be a minimal deliberate command).

## Acceptance Criteria

- A new user can get to “daemon connected” with one command (`rn daemon up`).
- `rn daemon status` answers “am I connected + attached + primary?” quickly.
- Duplicate/double-start is safe (idempotent “up”).

