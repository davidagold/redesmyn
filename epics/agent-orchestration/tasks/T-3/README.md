# T-3 Runner + CLI: spawn/attach/stop harness sessions (tmux-first)

## Metadata

```yaml
id: T-3
stacked_on: T-2
node:
  branch: rn/agent-orchestration/T-3-runner-tmux-cli
```

## Brief (local)

- Implement runner-owned process lifecycle for agent sessions:
  - spawn a harness process in the node worktree with consistent env injection
  - capture logs/output
  - stop/restart gracefully
- Prefer tmux-backed detached sessions to support `rn agent attach`.
- Add ergonomic CLI commands for humans:
  - `rn agent run/attach/stop/logs`
- Establish a baseline “generic” harness runner suitable for early dogfooding:
  - run the harness in the node worktree
  - inject a `PATH` shim so `git` resolves to `rn git ...` (best-effort invariant enforcement)
  - emit clear warnings when shimming is unavailable or likely ineffective
  - optionally print/bootstrap cooperative guidance (e.g., skill-based or textual prelude)

## Acceptance Criteria

- `rn agent run --node <node> --harness <...> --detach` starts a session that can be attached later.
- `rn agent attach <...>` works when tmux is available; a fallback path exists when not.
- Runner updates session + agent state (status, last-seen) in the control plane.
- The harness environment resolves `git` to the shim (or Redesmyn reports a degraded mode explicitly).
- Sessions reference a persisted `harness_profile_id` and record the resolved profile/attach metadata actually used.

## Updates

- Runner `git` shim strips itself from `PATH` before delegating to `rn git` to avoid recursion.
- Runner can “adopt” pre-existing worktree paths when the branch matches (useful when the node worktree already exists).
- Ad-hoc runs persist a `HarnessProfile` keyed by a stable hash of the resolved definition.
- Add a fleet workflow: `rn run --epic <slug> --fleet-size <n> [--harness …] [--detach]` to auto-assign tasks and start N sessions without per-task clicking.
- Include guardrails for `rn run`: `--dry-run`, skip tasks with active sessions by default, optional `--restart`, and clear “what will happen” output.
- Make `rn run` output ergonomic: compact table + “copy attach/logs” commands per started session.
